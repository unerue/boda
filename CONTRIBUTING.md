# Contributing to boda
---

## Code formatting and typing

### Formatting

To format your code, install `ufmt`

```bash
pip install ufmt==1.3.2 black==21.9b0 usort==0.6.4
```

```bash
ufmt format boda
```

### Type annotations

```bash
mypy --config-file mypy.ini
```

## Unit tests

```bash
pytest test -vvv
```

## Documentation
```bash
cd docs
make html-noplot
```