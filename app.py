from streamlit_app import run_app


def main():
    """
    Main entry point for the Streamlit application.

    This function initializes and runs the Streamlit app defined in `streamlit_app.py`.
    It includes basic error handling to catch and log any exceptions that occur during execution.
    """
    try:
        run_app()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()