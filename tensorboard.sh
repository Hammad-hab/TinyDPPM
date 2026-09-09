ls .tensorboard
status=$?

if [ $status -ne 0 ]; then
    echo "Creating tensorboard enviornment..."
    python3 -m venv ./.tensorboard && source ./.tensorboard/bin/activate && pip install tensorboard
fi

tensorboard --logdir runs