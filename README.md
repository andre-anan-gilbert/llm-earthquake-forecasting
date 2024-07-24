# llm-earthquake-forecasting

This project aims to develop a real-time earthquake monitoring web application using machine learning. The application will forecast earthquake magnitudes and depths by employing a CatBoost regression model trained on United States Geological Survey (USGS) earthquake data. Additionally, the project includes the integration of an LLM-powered AI agent designed to answer earthquake-related questions. This AI agent simulates the expertise of a USGS expert and incorporates various tools for earthquake information and forecasting to enhance its responses.

## Table of contents

- View [demo](https://drive.google.com/file/d/1mqNkPvWoyoslfvm98TLZzGBrlRcziHR6/view?usp=sharing).
- View [paper](/docs/paper/master.pdf).
- View [presentation](/docs/presentation.pdf).
- View [machine learning modeling](/ml/model.ipynb).

## Application

Interactive dashboard displaying recent earthquakes, with a map featuring earth-
quake forecasts and a corresponding table detailing the forecasted values for mag-
nitude and depth.

![dashboard](/docs/paper/img/dashboard.png)

Copilot interface showcasing human-AI collaboration, where an LLM-powered AI
agent assists by answering questions, running forecasts, and recommending pre-
cautionary actions.

![copilot](/docs/paper/img/copilot-forecast-example.png)

Explainable AI in action: The LLM-powered AI agent demonstrates its reasoning
process as it runs a forecast for California, USA. By following the reasoning steps,
the user can verify the accuracy and transparency of the LLM’s operations, thereby
building trust.

![copilot-reasoning](/docs/paper/img/explainable-ai-agent.png)

## Contributors

- Andre Anan Gilbert (3465546)
- Marc Grün (9603221)
- Felix Noll (9467152)
