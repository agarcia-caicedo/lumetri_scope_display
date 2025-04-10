from pathlib import Path

user_input = Path(input("Enter the path of your folder: "))
 
if not user_input.exists():
    raise ValueError(f"I did not find the file at {user_input}")
with open(user_input, "r+") as iof:
    print("Hooray we found your folder!")
