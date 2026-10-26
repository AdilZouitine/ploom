"""Check a built distribution from outside the source checkout."""

import importlib.metadata
import importlib.resources
import json

import ploom


def main() -> None:
    assert importlib.resources.files("ploom").joinpath("py.typed").is_file()
    assert ploom.check(lambda: None).executions == 1, "local scenarios need no extra"
    print(json.dumps({"version": importlib.metadata.version("ploom"), "file": ploom.__file__}))


if __name__ == "__main__":
    main()
