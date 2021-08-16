# Contributing

Contributions and bug fixes are always welcome, even if you're only reporting a bug.

!!! note "Legal"
    Contributors retain copyright on their substantial contributions. If applicable, when submitting a PR, please add yourself as an additional copyright holder in [LICENCE.md](https://github.com/virgesmith/neworder/LICENCE.md).

## Opening issues

If you find a bug or would like a new feature, please feel free to [open an issue](https://github.com/virgesmith/neworder/issues).

If you're taking the time to report a problem, even a seemingly minor one, it is appreciated, and a valid contribution to this project. Even better, if you can contribute by fixing bugs or adding features this is greatly appreciated.

See the [developer guide](./developer.md) to get a development environment set up.

## Contribution workflow

Here’s a quick guide for those unfamiliar with the contribution process:

1. [Fork this repository](https://github.com/virgesmith/neworder/fork) and then clone it locally:
```sh
git clone https://github.com/<your-github-handle>/neworder
```
2. Create a branch for your changes:
```sh
git checkout -b bug/fix-a-thing
# or
git checkout -b feature/add-a-thing
```
3. Create and commit a test that uses the new feature or illustrates the bug. It should fail:
```sh
pytest # fails
git commit -m <appropriate message>
```
4. Fix the bug or add the new feature and commit. Your test should now pass:
```sh
pytest # passes
git commit -m <appropriate message>
```
5. If all is well, push to your origin:
```sh
git push origin <your-branch-name>
```
6. If you're contributing new code, please add yourself as a copyright holder in the file [LICENCE.md](./licence.md) and commit this to your branch.
7. Finally, submit a [pull request.](https://help.github.com/articles/creating-a-pull-request)

