import traceback
from tabulate import tabulate

def main():
    try:
      print("""\t\t                  _____ _      
\t\t           /\\    / ____| |     
\t\t          /  \\  | (___ | |     
\t\t         / /\\ \\  \\___ \\| |     
\t\t        / ____ \\ ____) | |____ 
\t\t       /_/    \\_\\_____/|______|  \n\n""")
      data = [["Select mode:", "1. Train", "2. Evaluate", "3. Real-time Inference"]]
      print(tabulate(data, tablefmt="heavy_grid", colalign=("center", "center", "center")))
      choice = input("Enter choice (1/2/3): ").strip()

      if choice == '1':
          from src.models.train import train
          train()
      elif choice == '2':
          from src.models.evaluate import evaluate
          evaluate()
      elif choice == '3':
          from src.inference.realtime_inference import run_realtime
          run_realtime()
      else:
          print('Invalid choice. Please run again.')
    except KeyboardInterrupt:
      print("\nProcess interrupted. Exiting...")
    except Exception as e:
      tb = traceback.extract_tb(e.__traceback__)
      if tb:
          last_trace = tb[-1]
          print(f"Error on line {last_trace.lineno} in {last_trace.filename}: {e}")
      else:
          print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
