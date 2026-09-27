def test_optimizer_modules_import_without_solver_side_effects():
    import fplagent.optimizers.squad  # noqa: F401
    import fplagent.optimizers.transfers  # noqa: F401
    import fplagent.optimizers.captaincy  # noqa: F401
