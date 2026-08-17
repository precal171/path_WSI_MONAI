"""Command-line tools for data preparation, batch inference and test fixtures.

A package (rather than loose scripts) so library code can import from it. It is
deliberately NOT called `scripts`: each MONAI bundle has its own `scripts/`
package on sys.path, and two packages with the same name cannot coexist -- the
bundle would shadow this one.
"""
