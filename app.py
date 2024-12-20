import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt

# Load the pre-trained model
try:
    pipe = pkl.load(open('IPL_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please ensure `IPL_model.pkl` is in the same directory.")
    st.stop()

# Load the dataset for reference
try:
    df = pd.read_csv('final_dataset.csv').iloc[:, :-1]
except FileNotFoundError:
    st.error("Dataset file not found! Please ensure `final_dataset.csv` is in the same directory.")
    st.stop()

# Initialize variables from the dataset
Cities = sorted(df['City'].unique())
Teams = sorted(df['BattingTeam'].unique())

# Page title
st.title('🏏 IPL Win Predictor')

# Layout columns for team selection
col1, col2 = st.columns(2)

# Initialize session state for BattingTeam
if 'BattingTeam' not in st.session_state:
    st.session_state['BattingTeam'] = Teams[0]  # Default to the first team

# Batting Team Selection
BattingTeam = col1.selectbox(
    'Select Batting Team', 
    Teams, 
    index=Teams.index(st.session_state['BattingTeam']),
    help="Choose the team currently batting."
)

# Update available Bowling Teams
updated_teams = sorted(set(Teams).difference({BattingTeam}))
BowlingTeam = col2.selectbox(
    'Select Bowling Team', 
    updated_teams,
    help="Choose the opposing team."
)

# Update BattingTeam in session state
st.session_state['BattingTeam'] = BattingTeam

# City selection
city = st.selectbox('Select City', Cities, help="Choose the city where the match is being played.")

# Target input
Target = st.number_input(
    'Target', value=0, max_value=300, min_value=0, step=1, help="Enter the target score for the batting team."
)

# Layout columns for score and over/ball input
col3, col4, col5 = st.columns(3)

# Score input
Score = col3.number_input(
    'Score', value=0, max_value=Target, min_value=0, step=1, help="Enter the current score of the batting team."
)

# Overs and balls input
col4_1, col4_2 = col4.columns(2)
Over = col4_1.number_input(
    'Over Completed', value=0, max_value=20, min_value=0, step=1, help="Number of overs completed."
)
Ball = col4_2.number_input(
    'Ball Completed', value=1, max_value=5, min_value=0, step=1, help="Number of balls completed in the current over."
)

# Wickets input
Wickets = col5.number_input(
    'Wickets Gone', value=0, max_value=10, min_value=0, step=1, help="Number of wickets lost by the batting team."
)

# Prediction button
predict = st.button('Predict Probability')

# Prediction function
def score(input_df):
    result = pipe.predict_proba(input_df)
    return list(map(lambda x: round(x * 100), result[0]))

# Predict probability on button click
if predict:
    if Score > Target:
        st.error("Current score cannot exceed the target!")
    else:
        runs_left = Target - Score
        balls_left = 120 - (Over * 6 + Ball)
        wicket_left = 10 - Wickets
        crr = Score / ((Over * 6 + Ball) / 6) if (Over * 6 + Ball) > 0 else 0
        rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0

        # Input data for the model
        data = [[city, BattingTeam, BowlingTeam, runs_left, balls_left, wicket_left, Target, crr, rrr]]
        input_df = pd.DataFrame(data, columns=list(df.columns))

        # Make predictions
        try:
            loss, win = score(input_df)
        except ValueError:
            st.error("Error in model prediction. Check your input data.")
            st.stop()

        # Display predictions
        st.subheader("Win Probability")
        col6, col8 = st.columns(2)
        col6.metric(f"{BattingTeam}", f"{win}%")
        col8.metric(f"{BowlingTeam}", f"{loss}%")

        # Progress bar for visualization
        progress_bar = st.progress(0)
        for i in range(win + 1):
            progress_bar.progress(i)
            time.sleep(0.005)

        # Visualization: Pie chart
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        ax.pie(
            [win, loss], 
            labels=[BattingTeam, BowlingTeam], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#1f77b4', '#ff7f0e']
        )
        st.pyplot(fig)

        # Display input metrics
        st.subheader("Match Summary")
        st.table(
            input_df[['runs_left', 'balls_left', 'crr', 'rrr']]
        )