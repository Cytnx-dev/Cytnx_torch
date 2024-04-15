# cytnx-torch

# To develope:

[Setting up dev tools]

We use these two tools to manage the repo:
1. pre-commit tool: https://pre-commit.com

    To install, navigate to repo root folder:
    ```
    $cd cytnx_torch/
    $pre-commit install
    ```


2. rye: https://rye-up.com/

    * To set up the python enviroment, navigate to repo root folder

    ```
    $rye sync
    ```

    * To run test:
    ```
    $rye run pytest
    ```
