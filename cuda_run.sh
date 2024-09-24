# echo "Running Pythia14m with tinystories"
# python main.py --batch_size 16 --model_name "Pythia14m" --data "tinystories" --device "cuda"

# echo "Running Pythia70m with tinystories"
# python main.py --batch_size 16 --model_name "Pythia70m" --data "tinystories" --device "cuda"

# echo "Running Pythia160m with tinystories"
# python main.py --batch_size 16 --model_name "Pythia160m" --data "tinystories" --device "cuda"

echo "Running Pythia410m with tinystories"
python main.py --batch_size 16 --model_name "Pythia410m" --data "tinystories" --device "cuda"

echo "Running Pythia1b with tinystories"
python main.py --batch_size 16 --model_name "Pythia1b" --data "tinystories" --device "cuda"

echo "Running Pythia1.4b with tinystories"
python main.py --batch_size 16 --model_name "Pythia1.4b" --data "tinystories" --device "cuda"


# echo "Running Pythia14m with alpaca"
# python main.py --batch_size 16 --model_name "Pythia14m" --data "alpaca" --device "cuda"

# echo "Running Pythia70m with alpaca"
# python main.py --batch_size 16 --model_name "Pythia70m" --data "alpaca" --device "cuda"

# echo "Running Pythia160m with alpaca"
# python main.py --batch_size 16 --model_name "Pythia160m" --data "alpaca" --device "cuda"

# echo "Running Pythia410m with alpaca"
# python main.py --batch_size 16 --model_name "Pythia410m" --data "alpaca" --device "cuda"

# echo "Running Pythia1b with alpaca"
# python main.py --batch_size 16 --model_name "Pythia1b" --data "alpaca" --device "cuda"

# echo "Running Pythia1.4b with alpaca"
# python main.py --batch_size 16 --model_name "Pythia1.4b" --data "alpaca" --device "cuda"


echo "Running Pythia14m with summarisation"
python main.py --batch_size 16 --model_name "Pythia14m" --data "summarisation" --device "cuda"

echo "Running Pythia70m with summarisation"
python main.py --batch_size 16 --model_name "Pythia70m" --data "summarisation" --device "cuda"

echo "Running Pythia160m with summarisation"
python main.py --batch_size 16 --model_name "Pythia160m" --data "summarisation" --device "cuda"

echo "Running Pythia410m with summarisation"
python main.py --batch_size 16 --model_name "Pythia410m" --data "summarisation" --device "cuda"

echo "Running Pythia1b with summarisation"
python main.py --batch_size 16 --model_name "Pythia1b" --data "summarisation" --device "cuda"

echo "Running Pythia1.4b with summarisation"
python main.py --batch_size 16 --model_name "Pythia1.4b" --data "summarisation" --device "cuda"