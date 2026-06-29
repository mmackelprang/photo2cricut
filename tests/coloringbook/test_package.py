def test_package_imports_and_has_version():
    import photo2coloringbook as cb
    assert isinstance(cb.__version__, str) and cb.__version__
