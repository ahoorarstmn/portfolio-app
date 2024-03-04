import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model
import pickle

def load_and_preprocess_data():
    data = pd.read_csv("data.csv", parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    data.dropna(inplace=True)
    data['monthly_return'] = data['SPX'].pct_change().ffill()
    data['excess_return'] = data['monthly_return'] - (data['GS1M'] / 100)
    for lag in range(1, 13):
        data[f'lag_{lag}_return'] = data['excess_return'].shift(lag)
    data['future_excess_return'] = data['excess_return'].shift(-1)
    data['rolling_variance'] = data['monthly_return'].rolling(4).var()
    data.dropna(inplace=True)
    return data

def split_data(data):
    split_index = int(len(data) * 0.9)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def setup_ui():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.title("Portfolio Optimisation Widget")
    st.sidebar.image('fathom-logo.png', width=300)
    st.sidebar.markdown("Discover future investment potentials with our Financial Forecasting Application.")
    gamma = st.sidebar.slider('Risk Aversion', min_value=1, max_value=10, value=5)
    risk_profile = "Aggressive" if gamma <= 3.3 else "Moderate Risk Tolerance" if gamma <= 6.6 else "Conservative"
    st.sidebar.warning(risk_profile)
    selected_model = st.sidebar.selectbox('Optimisation Model', options=('Random Forest', 'Support Vector Machine', 'LightGBM'))
    return gamma, selected_model

def model_prediction(selected_model, X_test):
    if selected_model == 'Random Forest':
        with open('rf.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            pred = loaded_model.predict(X_test)
    else:
        st.sidebar.write('Not Available')
        
    return pd.Series(pred)



def calculate_portfolio_metrics(data, predicted_returns_series, gamma, split_index):
  garch = arch_model(data['monthly_return'], vol='Garch', p=1, q=1)
  garch_model = garch.fit(last_obs=split_index, disp='off')

  monthly_optimal_allocations = []
  monthly_sharpe_ratios = []
  monthly_max_drawdowns = []
  monthly_cumulative_returns = []

  for i in range(len(predicted_returns_series)):
    forecast = garch_model.forecast(horizon=1, start=split_index + i)
    forecasted_variance = forecast.variance.iloc[-1].values[0]

    # Calculate monthly optimal allocation for each iteration
    monthly_optimal_allocation = (1 / gamma) * (predicted_returns_series[i] / forecasted_variance)
    monthly_optimal_allocations.append(monthly_optimal_allocation)

    adjusted_predicted_returns = monthly_optimal_allocation * predicted_returns_series[i]
    risk_free_rate = data['GS1M'].iloc[split_index + i] / 100
    adjusted_returns = adjusted_predicted_returns - risk_free_rate

    # Sharpe Ratio Calculation for the month
    monthly_sharpe_ratio = np.mean(adjusted_returns) / np.std(adjusted_returns) if np.std(adjusted_returns) != 0 else 0
    monthly_sharpe_ratios.append(monthly_sharpe_ratio)

    cumulative_returns_series = pd.Series((1 + adjusted_returns).cumprod())
    rolling_max = cumulative_returns_series.rolling(window=252, min_periods=1).max()
    daily_drawdown = cumulative_returns_series / rolling_max - 1
    monthly_max_drawdown = daily_drawdown.min()
    monthly_max_drawdowns.append(monthly_max_drawdown)

    monthly_cumulative_return = cumulative_returns_series.iloc[-1] - 1
    monthly_cumulative_returns.append(monthly_cumulative_return)


  return monthly_optimal_allocations, monthly_sharpe_ratios, monthly_max_drawdowns, monthly_cumulative_returns




def display_charts(plot_data):
    with st.expander("Stats", expanded=True):
        a, b = st.columns(2)
        with a:
            with st.container():
                fig_a = px.line(plot_data, x='Date', y='SPX', title='S&P 500 Monthly')
                st.plotly_chart(fig_a, use_container_width=True)
        with b:
            with st.container():
                fig_b = go.Figure()
                fig_b.add_traces([
                    go.Scatter(x=plot_data['Date'], y=plot_data['monthly_return'], fill='tozeroy', fillcolor='blue', line_color='blue', name='Positive Returns'),
                    go.Scatter(x=plot_data['Date'], y=plot_data['monthly_return'].clip(upper=0), fill='tozeroy', fillcolor='red', line_color='red', name='Negative Returns')
                ])
                fig_b.update_layout(title='S&P Monthly Returns', xaxis_title='Date', yaxis_title='Return')
                st.plotly_chart(fig_b, use_container_width=True)

        c, d = st.columns(2)
        with c:
            with st.container():
                fig_c = px.histogram(plot_data, x='monthly_return', nbins=50, title='Distribution of Return Values', color_discrete_sequence=['#20B2AA'])
                st.plotly_chart(fig_c, use_container_width=True)
        with d:
            with st.container():
                fig_d = px.box(plot_data, y='monthly_return', title='Box Plot of Monthly Returns')
                st.plotly_chart(fig_d, use_container_width=True)

def display_portfolio_metrics(optimal_allocation, sharpe_ratio, max_drawdown, cumulative_return):
    f, g, h, i = st.columns(4)
    with f:
        with st.container(border=True):
            st.metric(label="Optimal Allocation", value=f'{optimal_allocation * 100:.2f}%', delta_color="inverse")
    with g:
        with st.container(border=True):
            st.metric(label="Sharpe Ratio", value=f'{sharpe_ratio:.2f}', delta_color="inverse")
    with h:
        with st.container(border=True):
            st.metric(label="Max Drawdown", value=f'{max_drawdown * 100:.2f}%', delta_color="inverse")
    with i:
        with st.container(border=True):
            st.metric(label="Cumulative Return", value=f'{cumulative_return * 100:.2f}%', delta_color="inverse")

def display_portfolio_metrics(monthly_optimal_allocations, monthly_sharpe_ratios, monthly_max_drawdowns, monthly_cumulative_returns, test_data):

    metrics_df = pd.DataFrame({
        'Month': test_data['Date'],
        'Optimal Allocation': np.array(monthly_optimal_allocations) * 100,
        'Sharpe Ratio': monthly_sharpe_ratios,
        'Max Drawdown': np.array(monthly_max_drawdowns) * 100,
        'Cumulative Return': np.array(monthly_cumulative_returns) * 100
    })
    with st.expander('Portfolio Metrics'):
        st.dataframe(metrics_df, use_container_width=True)
        st.bar_chart(monthly_cumulative_returns)
    
def main():
    data = load_and_preprocess_data()
    train_data, test_data = split_data(data)
    X_test = test_data[[f'lag_{lag}_return' for lag in range(1, 13)]]
    gamma, selected_model = setup_ui()
    predicted_returns_series = model_prediction(selected_model, X_test)
    optimal_allocation, sharpe_ratio, max_drawdown, cumulative_return = calculate_portfolio_metrics(data, predicted_returns_series, gamma, len(train_data))

    display_charts(data) 
    display_portfolio_metrics(optimal_allocation, sharpe_ratio, max_drawdown, cumulative_return, test_data) 
    
if __name__ == "__main__":
    main()


