import click
import os.path
from typing import Literal
from time import perf_counter

import nbformat
from nbconvert import PDFExporter, HTMLExporter

AllowedModes = Literal["html", "pdf"]

formats = {
    "pdf": {"exporter": PDFExporter, "mode": "wb"},
    "html": {"exporter": HTMLExporter, "mode": "w"},
}


@click.command()
@click.option("-n", "--notebook_file", required=True, help="Notebook file to export")
@click.option("-f", "--format", default="pdf", help="Export format. Only \"pdf\" or \"html\"")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output. Default is False")
def main(notebook_file: str, format: AllowedModes = "pdf", verbose: bool = False) -> None:
    assert format in AllowedModes.__args__, f"Format output must be one of {AllowedModes.__args__}"
    assert os.path.exists(notebook_file), f"Notebook file {notebook_file} does not exist"

    exporter = formats[format]["exporter"]()
    folder = os.path.dirname(notebook_file)
    basename = os.path.basename(notebook_file)
    filename, extension = os.path.splitext(basename)

    assert extension == ".ipynb", "Only .ipynb files are supported"

    if verbose:
        click.echo(f"Reading notebook file {notebook_file}...")
        t0 = perf_counter()

    with open(notebook_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    if verbose:
        click.echo(f"Notebook read in {perf_counter() - t0:.5f} seconds\n")

        click.echo(f"Exporting notebook to {format}...")
        t0 = perf_counter()

    body, resources = exporter.from_notebook_node(nb)
    output_file = os.path.join(folder, f"{filename}.{format}")
    with open(output_file, formats[format]["mode"]) as f:
        f.write(body)

    if verbose:
        click.echo(f"Notebook exported to: {perf_counter() - t0:.5f} seconds")

    click.echo(f"Notebook exported to {output_file}")


if __name__ == "__main__":
    main()
