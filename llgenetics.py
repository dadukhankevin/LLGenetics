from typing import Callable, List, Optional
import anthropic
from Finch.genetics import Individual
from Finch.layers.layer import Layer
from Finch.environments import Sequential
from Finch.genepools import GenePool
import random
import nltk
from nltk.corpus import words, wordnet
import numpy as np

nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Default system prompts
DEFAULT_MUTATION_SYSTEM_PROMPT = "You are an AI assistant that mutates text based on the given mutation amount and level. Output only what you are asked to, DO NOT EXPLAIN WHAT YOU ARE OUTPUTING OR PREFACE IT IN ANY WAY (:"
DEFAULT_CROSSOVER_SYSTEM_PROMPT = "You are an AI assistant that combines two texts to generate a child text. Output only what you are asked to, DO NOT EXPLAIN WHAT YOU ARE OUTPUTING OR PREFACE IT IN ANY WAY (:"
DEFAULT_POPULATE_SYSTEM_PROMPT = "You are an AI assistant that generates random prompts. Output only what you are asked to, DO NOT EXPLAIN WHAT YOU ARE OUTPUTING OR PREFACE IT IN ANY WAY (:"

# Prompt templates
MUTATION_PROMPT = "Mutate this text with {mutation_amount} changes at the {mutation_level} level: {text}"
CROSSOVER_PROMPT = "Combine the following two texts to generate a child text:\nText 1: {text1}\nText 2: {text2}"
POPULATE_PROMPT = "Generate a random prompt."


class LLM:
    def __init__(self, system_prompt: str, template: str, api_key: str, temp=.7) -> None:
        self.system_prompt = system_prompt
        self.template = template
        self.api_key = api_key
        self.temp = temp

    def apply(self, text: str):
        pass


class Claude(LLM):
    def __init__(self, system_prompt: str, template: str, api_key: str, temp=.7) -> None:
        super().__init__(system_prompt, template, api_key, temp)
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
        )

    def apply(self, text: str):
        message = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=self.temp,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.template.format(text=text)
                        }
                    ]
                }
            ]
        )
        return message.content[0].text


class LLMGenePool(GenePool):
    def __init__(self, llm: LLM, populate_prompt: str, system_prompt: Optional[str] = None):
        super().__init__(length=1)
        self.llm = llm
        self.populate_prompt = populate_prompt
        self.system_prompt = system_prompt or DEFAULT_POPULATE_SYSTEM_PROMPT

    def generate_individual(self):
        self.llm.system_prompt = self.system_prompt
        prompt = self.llm.apply(self.populate_prompt)
        new_individual = Individual(gene_pool=self, genes=np.array(list(prompt)), fitness=0)
        return new_individual


class PopulateLLM(Layer):
    def __init__(self, gene_pool: LLMGenePool, population_cap: int):
        super().__init__(individual_selection=None)
        self.gene_pool = gene_pool
        self.population_cap = population_cap

    def execute(self, individuals: List[Individual]) -> List[Individual]:
        while len(individuals) < self.population_cap:
            new_individual = self.gene_pool.generate_individual()
            new_individual.fitness = self.environment.fitness_function(new_individual)
            individuals.append(new_individual)
        return individuals


class MutationLayer(Layer):
    def __init__(self, llm: LLM, individual_selection: Callable[[List[Individual]], List[Individual]],
                 mutation_amount: str = "small", mutation_level: str = "word",
                 mutation_prompt: str = MUTATION_PROMPT, system_prompt: Optional[str] = DEFAULT_MUTATION_SYSTEM_PROMPT, refit=True):
        super().__init__(individual_selection=individual_selection, refit=refit)
        self.llm = llm
        self.mutation_amount = mutation_amount
        self.mutation_level = mutation_level
        self.mutation_prompt = mutation_prompt
        self.system_prompt = system_prompt or DEFAULT_MUTATION_SYSTEM_PROMPT

    def execute(self, individuals: List[Individual]) -> List[Individual]:
        self.llm.system_prompt = self.system_prompt
        selected_individuals = self.individual_selection(individuals)
        for individual in selected_individuals:
            text = "".join(individual.genes.astype(str))
            mutated_text = self.llm.apply(
                self.mutation_prompt.format(mutation_amount=self.mutation_amount, mutation_level=self.mutation_level,
                                            text=text))
            individual.genes = np.array(list(mutated_text))
        return individuals


class CrossoverLayer(Layer):
    def __init__(self, llm: LLM, individual_selection: Callable[[List[Individual]], List[Individual]],
                 crossover_prompt: str = CROSSOVER_PROMPT, system_prompt: Optional[str] = None, refit=True):
        super().__init__(individual_selection=individual_selection, refit=refit)
        self.llm = llm
        self.crossover_prompt = crossover_prompt
        self.system_prompt = system_prompt or DEFAULT_CROSSOVER_SYSTEM_PROMPT

    def execute(self, individuals: List[Individual]) -> List[Individual]:
        self.llm.system_prompt = self.system_prompt
        selected_individuals = self.individual_selection(individuals)
        for i in range(0, len(selected_individuals) - 1, 2):
            text1 = "".join(selected_individuals[i].genes.astype(str))
            text2 = "".join(selected_individuals[i + 1].genes.astype(str))
            crossover_text = self.llm.apply(self.crossover_prompt.format(text1=text1, text2=text2))
            selected_individuals[i].genes = np.array(list(crossover_text))
        return individuals


class SortByFitnessLayer(Layer):
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        individuals.sort(key=lambda individual: individual.fitness, reverse=True)
        return individuals


class CapPopulationLayer(Layer):
    def __init__(self, max_population: int):
        super().__init__(individual_selection=None)
        self.max_population = max_population

    def execute(self, individuals: List[Individual]) -> List[Individual]:
        if len(individuals) > self.max_population:
            individuals = individuals[:self.max_population]
        return individuals


### PURE TEXT LAYERS


class EnglishPool(GenePool):
    def __init__(self, length):
        super().__init__(length)
        self.vocab = words.words()

    def generate_individual(self):
        words = random.choices(self.vocab, k=self.length)
        new_individual = Individual(gene_pool=self, genes=np.array([word + " " for word in words]), fitness=0)
        return new_individual
class BasicTextMutation(Layer):
    def __init__(self, gene_pool, individual_selection, insertion_rate, deletion_rate, substitution_rate, refit=True):
        super().__init__(individual_selection=individual_selection, refit=refit)
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.substitution_rate = substitution_rate
        self.gene_pool = gene_pool

    def execute(self, individuals):
        selected_individuals = self.individual_selection(individuals)
        for individual in selected_individuals:
            mutated_genes = individual.genes.copy()
            new_mutated_genes = []
            for i, gene in enumerate(mutated_genes):
                if random.random() < self.insertion_rate:
                    new_mutated_genes.append(random.choice(self.gene_pool.vocab) + " ")
                if random.random() >= self.deletion_rate:
                    new_mutated_genes.append(gene)
                if random.random() < self.substitution_rate:
                    new_mutated_genes.append(random.choice(self.gene_pool.vocab)+ " ")
            individual.genes = np.array(new_mutated_genes)
        return individuals

class AdvancedTextMutation(Layer):
    def __init__(self, individual_selection, mutation_rate=0.1, refit=True):
        super().__init__(individual_selection=individual_selection, refit=refit)
        self.mutation_rate = mutation_rate

    def execute(self, individuals):
        selected_individuals = self.individual_selection(individuals)
        for individual in selected_individuals:
            mutated_genes = []
            for word in individual.genes:
                if random.random() < self.mutation_rate:
                    pos_tags = nltk.pos_tag([word.item()])
                    pos = pos_tags[0][1]
                    if pos.startswith('NN'):  # Noun
                        mutation_type = random.choice(["pluralize", "singularize", "synonym"])
                        if mutation_type == "pluralize":
                            mutated_genes.append(nltk.pluralize(word.item()))
                        elif mutation_type == "singularize":
                            mutated_genes.append(nltk.singularize(word.item()))
                        else:
                            synsets = wordnet.synsets(word.item())
                            if synsets:
                                lemmas = synsets[0].lemmas()
                                if lemmas:
                                    mutated_genes.append(lemmas[0].name())
                                else:
                                    mutated_genes.append(word.item())
                            else:
                                mutated_genes.append(word.item())
                    elif pos.startswith('VB'):  # Verb
                        mutation_type = random.choice(["present", "past", "future", "continuous"])
                        if mutation_type == "present":
                            mutated_genes.append(word.item())
                        elif mutation_type == "past":
                            mutated_genes.append(nltk.stem.WordNetLemmatizer().lemmatize(word.item(), 'v'))
                        elif mutation_type == "future":
                            mutated_genes.append("will " + word.item())
                        else:
                            mutated_genes.append(nltk.stem.WordNetLemmatizer().lemmatize(word.item(), 'v') + "ing")
                    elif pos.startswith('JJ') or pos.startswith('RB'):  # Adjective or Adverb
                        mutation_type = random.choice(["comparative", "superlative", "antonym"])
                        if mutation_type == "comparative":
                            mutated_genes.append(nltk.stem.WordNetLemmatizer().lemmatize(word.item(), 'a') + "er")
                        elif mutation_type == "superlative":
                            mutated_genes.append(nltk.stem.WordNetLemmatizer().lemmatize(word.item(), 'a') + "est")
                        else:
                            synsets = wordnet.synsets(word.item())
                            if synsets:
                                antonyms = []
                                for synset in synsets:
                                    for lemma in synset.lemmas():
                                        if lemma.antonyms():
                                            antonyms.append(lemma.antonyms()[0].name())
                                if antonyms:
                                    mutated_genes.append(random.choice(antonyms))
                                else:
                                    mutated_genes.append(word.item())
                            else:
                                mutated_genes.append(word.item())
                    else:
                        mutated_genes.append(word.item())
                else:
                    mutated_genes.append(word.item())
            individual.genes = np.array(mutated_genes)
        return individuals