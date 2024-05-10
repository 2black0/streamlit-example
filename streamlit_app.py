import streamlit as st

def main():
    st.title('Simple Streamlit App')
    
    # Create a text input box
    user_input = st.text_input("Enter some text")

    # Display the input text back to the user
    if user_input:
        st.write(f'You entered: {user_input}')

if __name__ == "__main__":
    main()