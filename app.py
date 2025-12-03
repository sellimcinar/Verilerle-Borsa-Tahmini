import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_loader
import datetime
import numpy as np
import pandas as pd

st.set_page_config(page_title="BIST AI Analyst - [Selim CINAR]", page_icon="🤖", layout="wide")

st.title("BIST100 & USD/TRY Analysis Dashboard")

# Sidebar for controls
st.sidebar.header("Settings")
period = st.sidebar.selectbox("Select Time Period", ["1y", "2y", "5y", "10y", "max"], index=2)
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30)

# Ticker Selection
default_tickers = ["XU100.IS"]
with st.spinner("Fetching BIST tickers..."):
    available_tickers = data_loader.get_all_bist_tickers()

# Ensure XU100.IS is in the list if not already
if "XU100.IS" not in available_tickers:
    available_tickers.append("XU100.IS")
    
# Sort tickers for better UX
available_tickers.sort()

selected_tickers = st.sidebar.multiselect("Select Tickers", available_tickers, default=default_tickers)

# Ensure TRY=X is always loaded for comparison
load_tickers = list(set(selected_tickers + ["TRY=X"]))

# Load Data
with st.spinner("Fetching data..."):
    data = data_loader.load_data(load_tickers, period=period)

# Main Dashboard
if not selected_tickers:
    st.warning("Please select at least one ticker.")
else:
    # Create tabs for each selected ticker
    tabs = st.tabs(selected_tickers)
    
    for ticker in selected_tickers:
        with tabs[selected_tickers.index(ticker)]:
            st.header(f"{ticker} Analysis")
            
            # --- Executive Scorecard ---
            st.subheader("📊 Executive Scorecard")
            score_cols = st.columns(4)
            
            # 1. Current Price
            if isinstance(data.columns, pd.MultiIndex):
                last_price = data[ticker]['Close'].iloc[-1]
                prev_price = data[ticker]['Close'].iloc[-2]
            else:
                last_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                
            daily_ret = (last_price - prev_price) / prev_price
            score_cols[0].metric("Current Price", f"{last_price:.2f} TL", f"{daily_ret*100:.2f}%")
            
            # 2. AI Forecast (30 Days)
            with st.spinner("Calculating Scorecard..."):
                forecast = data_loader.forecast_prophet(data, ticker, days=30)
                if not forecast.empty:
                    future_price = forecast['yhat'].iloc[-1]
                    exp_return = (future_price - last_price) / last_price
                    score_cols[1].metric("AI Forecast (30d)", f"{future_price:.2f} TL", f"{exp_return*100:.2f}% Expected")
                else:
                    score_cols[1].metric("AI Forecast (30d)", "N/A", "Insufficient Data")
                    
                # 3. Risk Level (VaR)
                mc_data = data_loader.run_monte_carlo(ticker)
                if mc_data:
                    var_95 = mc_data['metrics']['var_95']
                    var_pct = (var_95 / last_price) - 1
                    risk_label = "High" if var_pct < -0.10 else "Medium" if var_pct < -0.05 else "Low"
                    score_cols[2].metric("Risk Level (VaR 95%)", f"{risk_label}", f"{var_pct*100:.2f}% Risk")
                else:
                    score_cols[2].metric("Risk Level", "N/A", "Insufficient Data")
                    
                # 4. Fundamental Score
                fund_data = data_loader.get_fundamental_metrics(ticker)
                if fund_data:
                    avg_score = np.mean(list(fund_data['scores'].values()))
                    score_cols[3].metric("Fundamental Quality", f"{avg_score:.0f}/100", "Based on 5 Metrics")
                else:
                    score_cols[3].metric("Fundamental Quality", "N/A", "No Data")
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{ticker} Analysis")
                stock_df = data_loader.calculate_metrics(data, ticker)
                
                # Main Chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], name='Close'), secondary_y=False)
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_50'], name='SMA 50'), secondary_y=False)
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_200'], name='SMA 200'), secondary_y=False)
                fig.update_layout(title=f"{ticker} Price & Moving Averages", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")
                
                # Forecast
                if st.checkbox(f"Show {ticker} Forecast", key=f"forecast_{ticker}"):
                    with st.spinner("Forecasting..."):
                        # Forecast is already calculated for scorecard, but maybe for different days?
                        # Scorecard uses 30 days. Slider uses forecast_days.
                        # If forecast_days == 30, we could reuse, but simpler to re-run or use cache.
                        forecast = data_loader.forecast_prophet(data, ticker, days=forecast_days)
                        
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(width=0)))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(width=0), fill='tonexty'))
                        fig_forecast.update_layout(title=f"{ticker} {forecast_days}-Day Forecast", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig_forecast, use_container_width=True, key=f"prophet_{ticker}")

                # Pattern Matcher
                st.markdown("### 🧬 Historical Pattern Matcher")
                if st.button(f"Find Similar Patterns for {ticker}", key=f"pattern_btn_{ticker}"):
                    with st.spinner(f"Scanning BIST30 history for patterns similar to {ticker}..."):
                        matches = data_loader.find_similar_patterns(ticker)
                        
                        if not matches:
                            st.warning("No sufficient data found for pattern matching.")
                        else:
                            # Prepare data for plotting
                            # Get current stock last 60 days for context
                            end_date = datetime.datetime.now()
                            start_date = end_date - datetime.timedelta(days=120) 
                            # We need raw data for pattern matching visualization
                            # data_loader.load_data is cached, so calling it again is cheap if args match
                            # But here we need specific ticker data
                            current_data = data_loader.load_data([ticker], period="1y") 
                            
                            if isinstance(current_data.columns, pd.MultiIndex):
                                current_series = current_data[ticker]['Close']
                            else:
                                current_series = current_data['Close']
                                
                            current_window = current_series.tail(60)
                            last_date = current_window.index[-1]
                            last_price_pm = current_window.iloc[-1]
                            
                            # Plot
                            fig_pattern = go.Figure()
                            
                            # 1. Current Stock (Last 60 days)
                            fig_pattern.add_trace(go.Scatter(
                                x=list(range(60)), 
                                y=current_window.values, 
                                name=f"Current {ticker}",
                                line=dict(color='blue', width=3)
                            ))
                            
                            # 2. Top 3 Matches Projections
                            colors = ['red', 'orange', 'green']
                            future_returns_list = []
                            
                            for idx, match in enumerate(matches):
                                score_pct = match['score'] * 100
                                match_ticker = match['ticker']
                                match_date = match['date'].strftime('%Y-%m-%d')
                                
                                # Project future returns onto current price
                                future_returns = match['future_returns']
                                future_returns_list.append(future_returns)
                                projected_prices = last_price_pm * (1 + future_returns)
                                
                                # X-axis for future: 60 to 69
                                future_x = list(range(59, 59 + len(projected_prices)))
                                
                                fig_pattern.add_trace(go.Scatter(
                                    x=future_x,
                                    y=projected_prices,
                                    name=f"Match #{idx+1}: {match_ticker} ({match_date}) - {score_pct:.1f}%",
                                    line=dict(color=colors[idx], dash='dot', width=1)
                                ))
                            
                            # 3. Composite Projection (Average)
                            if future_returns_list:
                                avg_future_returns = np.mean(future_returns_list, axis=0)
                                avg_projected_prices = last_price_pm * (1 + avg_future_returns)
                                future_x = list(range(59, 59 + len(avg_projected_prices)))
                                
                                fig_pattern.add_trace(go.Scatter(
                                    x=future_x,
                                    y=avg_projected_prices,
                                    name="Average Projection",
                                    line=dict(color='orange', width=4, dash='dash')
                                ))
                                
                                # Signal Generation
                                avg_final_return = avg_future_returns[-1]
                                if avg_final_return > 0.02:
                                    signal = "BULLISH (Yükseliş Beklentisi) 🚀"
                                    signal_color = "green"
                                elif avg_final_return < -0.02:
                                    signal = "BEARISH (Düşüş Beklentisi) 🔻"
                                    signal_color = "red"
                                else:
                                    signal = "NEUTRAL 😐"
                                    signal_color = "gray"
                                    
                                # Confidence Metric
                                directions = [ret[-1] > 0 for ret in future_returns_list]
                                if all(directions) or not any(directions):
                                    confidence = "High 🔥"
                                else:
                                    confidence = "Low ⚠️"
                                
                                st.markdown(f"### Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Confidence:** {confidence}")

                            fig_pattern.update_layout(
                                title=f"Top 3 Historical Pattern Matches & Projections",
                                xaxis_title="Days (0-59: History, 60+: Projection)",
                                yaxis_title="Price",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_pattern, use_container_width=True, key=f"pattern_chart_{ticker}")
                            
                            # Display Metrics
                            cols = st.columns(3)
                            for idx, match in enumerate(matches):
                                with cols[idx]:
                                    avg_return = np.mean(match['future_returns']) * 100
                                    st.metric(
                                        label=f"Match #{idx+1} Return (10d)",
                                        value=f"{avg_return:.2f}%",
                                        delta=f"{match['score']*100:.1f}% Similarity"
                                    )

                # Risk Analysis (Monte Carlo)
                st.markdown("### 🎲 Risk Analysis (Monte Carlo Simulation)")
                if st.button(f"Run Monte Carlo Simulation for {ticker}", key=f"mc_btn_{ticker}"):
                    with st.spinner(f"Running 1000 simulations for {ticker}..."):
                        mc_data = data_loader.run_monte_carlo(ticker)
                        
                        if mc_data:
                            paths = mc_data['paths']
                            metrics = mc_data['metrics']
                            
                            # Plot Spaghetti Chart
                            fig_mc = go.Figure()
                            
                            days = paths.shape[0]
                            x_axis = list(range(days))
                            
                            for i in range(min(1000, paths.shape[1])):
                                fig_mc.add_trace(go.Scatter(
                                    x=x_axis,
                                    y=paths[:, i],
                                    mode='lines',
                                    line=dict(color='blue', width=1),
                                    opacity=0.05,
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                            fig_mc.add_trace(go.Scatter(
                                x=[days-1],
                                y=[metrics['var_95']],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name='VaR 95% (Worst Case)'
                            ))
                            
                            fig_mc.add_trace(go.Scatter(
                                x=[days-1],
                                y=[metrics['upside_95']],
                                mode='markers',
                                marker=dict(color='green', size=10),
                                name='Upside 95% (Best Case)'
                            ))
                            
                            fig_mc.add_trace(go.Scatter(
                                x=[days-1],
                                y=[metrics['expected_price']],
                                mode='markers',
                                marker=dict(color='orange', size=10),
                                name='Expected Price'
                            ))
                            
                            fig_mc.update_layout(
                                title=f"Monte Carlo Simulation (1000 Iterations) - 30 Days Forecast",
                                xaxis_title="Days",
                                yaxis_title="Price",
                                showlegend=True
                            )
                            st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_chart_{ticker}")
                            
                            mc_cols = st.columns(3)
                            mc_cols[0].metric("Expected Price", f"{metrics['expected_price']:.2f}", f"{(metrics['expected_price']/metrics['last_price']-1)*100:.2f}%")
                            mc_cols[1].metric("VaR 95% (Risk)", f"{metrics['var_95']:.2f}", f"{(metrics['var_95']/metrics['last_price']-1)*100:.2f}%")
                            mc_cols[2].metric("Potential Upside (95%)", f"{metrics['upside_95']:.2f}", f"{(metrics['upside_95']/metrics['last_price']-1)*100:.2f}%")
                        else:
                            st.error("Could not run simulation. Insufficient data.")

                # Fundamental Health Radar
                st.markdown("### 🧭 Fundamental Health Check")
                if st.checkbox(f"Show Fundamental Radar for {ticker}", key=f"radar_chk_{ticker}"):
                    with st.spinner(f"Fetching fundamentals for {ticker}..."):
                        fund_data = data_loader.get_fundamental_metrics(ticker)
                        
                        if fund_data:
                            scores = fund_data['scores']
                            raw = fund_data['raw']
                            
                            categories = list(scores.keys())
                            values = list(scores.values())
                            
                            categories.append(categories[0])
                            values.append(values[0])
                            
                            fig_radar = go.Figure()
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=ticker,
                                fillcolor='rgba(255, 215, 0, 0.5)',
                                line=dict(color='gold')
                            ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )
                                ),
                                showlegend=False,
                                title=f"Fundamental Health Score (0-100)"
                            )
                            
                            col_radar1, col_radar2 = st.columns([2, 1])
                            
                            with col_radar1:
                                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_chart_{ticker}")
                                
                            with col_radar2:
                                st.markdown("#### Raw Metrics")
                                st.dataframe(pd.DataFrame.from_dict(raw, orient='index', columns=['Value']))
                        else:
                            st.warning("Could not fetch fundamental data.")

            with col2:
                st.subheader("USD/TRY (TRY=X)")
                try_df = data_loader.calculate_metrics(data, "TRY=X")
                
                # Plotly Chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=try_df.index, y=try_df['Close'], name='Close', line=dict(color='green')))
                fig2.add_trace(go.Scatter(x=try_df.index, y=try_df['SMA_50'], name='SMA 50'))
                fig2.add_trace(go.Scatter(x=try_df.index, y=try_df['SMA_200'], name='SMA 200'))
                fig2.update_layout(title="USD/TRY Exchange Rate", xaxis_title="Date", yaxis_title="Rate")
                st.plotly_chart(fig2, use_container_width=True, key=f"try_chart_{ticker}")

            # Correlation
            st.subheader("Analysis")
            correlation = data_loader.calculate_correlation(data, ticker, "TRY=X")
            st.metric(f"Correlation ({ticker} vs USD/TRY)", f"{correlation:.4f}")
