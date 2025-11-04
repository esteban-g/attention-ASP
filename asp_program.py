import clingo
import os

# Function to read a file and return its contents as a list of terms
def read_vocabulary(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Define the base ASP program as a template
base_program = """
a :- b, c, not d.
b.
c.
"""

# Define the external files for each atom
vocabulary_files = {
    'b': 'art_vocab.txt',
    'c': 'finance_vocab.txt',
    # Add more atoms and files as needed
}

# Read the vocabularies and generate the ASP rules
additional_rules = []
for atom, file_path in vocabulary_files.items():
    if os.path.exists(file_path):
        terms = read_vocabulary(file_path)
        if terms:
            # Use the first term as the atom value
            additional_rules.append(f"{atom}({terms[0]}).")

# Combine the base program and the additional rules
full_program = base_program + "\n" + "\n".join(additional_rules)

# Use Clingo to solve the program
ctl = clingo.Control()
ctl.add("base", [], full_program)
ctl.ground([("base", [])])

# Solve and print the models
models = []
with ctl.solve(yield_=True) as handle:
    for model in handle:
        models.append(model.symbols(shown=True))

# Print the models
for i, model in enumerate(models, 1):
    print(f"Model {i}: {model}")
