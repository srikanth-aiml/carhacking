import os
import configparser

import typer

app = typer.Typer()

@app.command()
def extract_features(
    inilocation: str = typer.Option(
                os.getcwd(), "--iniloc", "-i", 
                prompt="Location of carhacking.ini"), 
) -> None:
    print(f"Begin executing Carhacking app step 1 ...")

    print(f"Finished executing Carhacking app step 1")


@app.command()
def train(
    inilocation: str = typer.Option(
                os.getcwd(), "--iniloc", "-i", 
                prompt="Location of carhacking.ini"), 
) -> None:
    print(f"Begin executing Carhacking app training ...")

    print(f"Finished executing Carhacking app training")



if __name__ == "__main__":
    app()