"""Runnable compatibility entry point for the SatFeed Streamlit dashboard."""

import runpy


# Streamlit reruns this file for widget changes; execute the dashboard each time
# instead of importing a cached module whose top-level UI code would be skipped.
runpy.run_module("src.ui.dashboard", run_name="__main__")