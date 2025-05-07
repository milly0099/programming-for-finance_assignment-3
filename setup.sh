#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
address = \"0.0.0.0\"\n\

[theme]\n\
primaryColor = \"#e50914\"\n\
backgroundColor = \"#121212\"\n\
secondaryBackgroundColor = \"#1e1e1e\"\n\
textColor = \"#f0f0f0\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml