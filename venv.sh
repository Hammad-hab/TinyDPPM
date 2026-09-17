if [ ! -d "bin" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .
    source ./bin/activate
    pip install -r requirements.txt
else
    source ./bin/activate
fi