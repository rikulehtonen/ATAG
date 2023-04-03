# ATAG

an experimental tool that combines reinforcement learning and test libraries to build tests. Current version is only used for demo.

![Demo](resources/material/atag_demo.gif)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage

- atag: ML algorithm
- browserenv: connection between the browser test library and the ML algorithm

Set self.targets, self.elements and self.actions in browserenv __init__.py file. Run the training using:

```bash
python run.py
```

## Contributing

This is a private repository for a master thesis project. Pull requests are not possible for the time being. Suggestions are always welcome: riku.lehtonen@tietoevry.com

## License

[MIT](https://choosealicense.com/licenses/mit/)