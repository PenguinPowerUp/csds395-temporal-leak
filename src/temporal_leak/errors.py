class LeakageError(AssertionError):
    """A pipeline used, or could use, information from after its scoring instant."""
