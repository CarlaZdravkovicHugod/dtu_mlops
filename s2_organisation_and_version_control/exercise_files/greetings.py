# uv run greetings.py            # should print "Hello World!"
# uv run greetings.py --count=3  # should print "Hello World!" three times
# uv run greetings.py --help     # should print the help message, informing the user of

import typer

def main(count: int = 1) -> None:
    """Print 'Hello World!' a specified number of times.
    """
    for _ in range(count):
        print("Hello World!") 
typer.run(main)

