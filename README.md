# :earth_americas: GDP dashboard template

A simple Streamlit app showing the GDP of different countries in the world.

source ~/Desktop/TPG/Reseau/test/.venv/bin/activate

# App 1
export APP_PASSWORD="foo123"
streamlit run app1.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true

# App 2
export APP_PASSWORD="bar456"
streamlit run app2.py \
  --server.address 0.0.0.0 \
  --server.port 8502 \
  --server.headless true

export APP_PASSWORD="mySecret123"
streamlit run app.py --server.address 0.0.0.0

python scripts/precompute_indicators.py \
  --input data/soustraitance.csv \
  --output-dir data


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
