import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="4-gram Language Model Visualizer", page_icon="🔤")

probs = np.load("dev/ngram_probs.npy")

chars = ["\n"] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def get_probs(context):
    indices = [char_to_idx[c] for c in context]
    return probs[tuple(indices)]


def sample_next_char(probs):
    return np.random.choice(chars, p=probs)


def plot_probabilities(context, display_area=None):
    next_probs = get_probs(context)

    sorted_indices = np.argsort(next_probs)[::-1]
    sorted_probs = next_probs[sorted_indices]
    sorted_chars = [chars[i] for i in sorted_indices]

    most_probable_char = sorted_chars[0] if sorted_chars[0] != "\n" else " "

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        sorted_probs.reshape(-1, 1),
        annot=True,
        cmap="viridis",
        cbar=True,
        yticklabels=sorted_chars,
        xticklabels=[""],
    )
    plt.title(
        f"""Next Character Probabilities for context: '{context}'
        Most Probable: '{most_probable_char}' ({sorted_probs[0]:.2f})"""
    )
    plt.ylabel("Characters")
    plt.xlabel("Probability")
    plt.xticks(rotation=45)

    if display_area is None:
        return plt

    display_area.pyplot(plt.gcf())
    plt.close()


def generate_text(seed, length, text_display_area, prob_display_area):
    generated_text = seed
    context = seed

    for _ in range(length):
        next_probs = get_probs(context)
        next_char = sample_next_char(next_probs)
        generated_text += next_char
        context = context[1:] + next_char

        if text_display_area and prob_display_area:
            text_display_area.text(generated_text)
            plot_probabilities(context, prob_display_area)
            time.sleep(1)

    return generated_text


def main():
    st.title("4-gram Language Model Visualizer")

    st.sidebar.header("Model Information")
    st.sidebar.write("Sequence Length: `4`")
    st.sidebar.write("Vocabulary Size:", len(chars))
    st.sidebar.write("Using pre-trained probabilities from `ngram_probs.npy`")

    page = st.selectbox(
        "Choose a page",
        ["Probability Visualization", "Text Generation"],
    )

    if page == "Probability Visualization":
        st.header("Probability Visualization")
        context = st.text_input(
            "Enter a context (3+ characters):",
            value="the",
            help="Enter a context of 3 or more characters to see the probability distribution for the next character (based on the 4-gram model).",
        )
        if len(context) < 3:
            st.warning("Please enter 3 characters or more for the context.")
        elif len(context) >= 3:
            context_trimmed = context[-3:]

            fig = plot_probabilities(context_trimmed)
            st.pyplot(fig)

    elif page == "Text Generation":
        st.header("Text Generation")
        seed = st.text_input("Enter a seed text (3 characters):", value="the")
        if len(seed) != 3:
            st.warning("Please enter exactly 3 characters for the seed.")
        else:
            length = st.slider("Select generation length:", 10, 200, 50)
            if st.button("Generate Text"):
                text_display_area = st.empty()
                prob_display_area = st.empty()
                generated_text = generate_text(
                    seed, length, text_display_area, prob_display_area
                )
                st.write("Generated Text:")
                st.write(generated_text)

        st.header("How It Works")
        st.write(
            """
            1.  The 4-gram model predicts the next character using the previous 3 characters as context.
            2.  It uses probabilities from 'ngram_probs.npy'.
            3.  A bar chart displays the probability distribution for the next character.
            4.  Characters are sampled based on these probabilities during text generation.
            5.  The process repeats with the last 3 characters as the new context.
            """
        )

        st.header("Experiment")
        st.write(
            """
            Try various to see how the model behaves:
            - Use common English trigrams like "the", "and", "ing" as seeds or contexts.
            """
        )


if __name__ == "__main__":
    main()
