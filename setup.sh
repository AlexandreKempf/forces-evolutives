mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"alexandre.kempf@cri-paris.org\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
