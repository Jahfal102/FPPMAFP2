
# Streamlit app
def main():
    st.title('COVID-19 Pneumonia Detection App')

    # Upload an image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image.", use_column_width=True)

        # Make predictions on the uploaded image
        predictions = predict(uploaded_file)

        if predictions is not None:
            st.subheader("Predictions:")
            # Assuming your model outputs probabilities for each class
            st.write(f"COVID-19 Probability: {predictions[0][0]:.2%}")
            st.write(f"Pneumonia Probability: {predictions[0][1]:.2%}")
            st.write(f"Normal Probability: {predictions[0][2]:.2%}")

            # Add confusion matrix and classification report
            true_label = "Normal" if "Normal" in uploaded_file.name else "COVID-19"
            predicted_label = "Normal" if predictions[0][2] > predictions[0][0] else "COVID-19"
            st.write(f"True Label: {true_label}")
            st.write(f"Predicted Label: {predicted_label}")

            # Display confusion matrix
            y_true = ["Normal", "COVID-19"]
            y_pred = [predicted_label, true_label]
            cm = confusion_matrix(y_true, y_pred, labels=["Normal", "COVID-19"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y_true, yticklabels=y_true)
            st.pyplot(plt)

if __name__ == "__main__":
    main()
