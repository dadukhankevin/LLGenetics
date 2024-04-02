# Large Language Genetics (LLGenetics)

LLGenetics is a Python library that combines large language models (LLMs) with genetic algorithms to evolve and optimize text prompts. It provides a set of customizable layers and components to build genetic algorithms specifically tailored for working with text data and LLMs.

## Features

- Integration with the Anthropic API for using the Claude model
- Customizable gene pools for generating initial text populations
- Mutation layers for introducing variations in the text
- Crossover layers for combining and generating new text offspring
- Fitness evaluation and selection layers
- Support for both LLM-based and pure text-based genetic operations

## Installation

To install LLGenetics, clone the repository from GitHub:

```git
git clone https://github.com/dadukhankevin/LLGenetics.git
```

## Usage

Here's a basic example of how to use LLGenetics:

```python
from llgenetics import Claude, LLMGenePool, PopulateLLM, MutationLayer, CrossoverLayer, SortByFitnessLayer, CapPopulationLayer, Sequential

# Initialize the LLM (Claude) with API key and prompt templates
llm = Claude(api_key="your_api_key", system_prompt=DEFAULT_MUTATION_SYSTEM_PROMPT, template=MUTATION_PROMPT)

# Create a gene pool for generating initial text population
gene_pool = LLMGenePool(llm=llm, populate_prompt=POPULATE_PROMPT)

# Define the layers of the genetic algorithm
layers = [
    PopulateLLM(gene_pool=gene_pool, population_cap=10),
    MutationLayer(llm=llm, individual_selection=lambda individuals: individuals[:5]),
    CrossoverLayer(llm=llm, individual_selection=lambda individuals: individuals[:5]),
    SortByFitnessLayer(),
    CapPopulationLayer(max_population=10)
]

# Create the genetic algorithm environment
environment = Sequential(layers=layers, fitness_function=your_fitness_function)

# Run the genetic algorithm for a specified number of generations
environment.run(generations=10)
```

## Customization

LLGenetics provides various customization options:

- Implement your own fitness function to evaluate the quality of the generated text prompts
- Adjust the mutation and crossover parameters to control the level of variation introduced
- Define custom selection strategies for choosing individuals for mutation and crossover
- Experiment with different prompt templates to optimize the text generation process
- Right now Claude Haiku is the only LLM fast and smart enough to be useful in GAs, but more will be added.

## Contributing

Contributions to LLGenetics are welcome! If you find any bugs, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request on the GitHub repository.

## License

LLGenetics is open-source and released under the [MIT License](https://opensource.org/licenses/MIT).