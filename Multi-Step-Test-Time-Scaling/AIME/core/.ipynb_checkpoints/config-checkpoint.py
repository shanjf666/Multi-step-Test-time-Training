"""
Configuration classes for TTT methods
"""

class Config:
    def __init__(self):
        self.system_prompt = (
            """
                Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
                Answer: ## Step1: There are 15 trees originally. ## Step2: Then there were 21 trees after some more were planted. ## Step3: So there must have been 21 - 15 = 6. The answer is 6.
                
                
                Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
                Answer: ## Step1: There are originally 3 cars. ## Step2: 2 more cars arrive. ## Step3: 3 + 2 = 5. The answer is 5.
                
                
                Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
                Answer: ## Step1: Originally, Leah had 32 chocolates. ## Step2: Her sister had 42. ## Step3: So in total they had 32 + 42 = 74. ## Step4: After eating 35, they had 74 - 35 = 39. The answer is 39.
                
                
                Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
                Answer: ## Step1: Jason started with 20 lollipops. ## Step2: Then he had 12 after giving some to Denny. ## Step3: So he gave Denny 20 - 12 = 8. The answer is 8.
                
                
                Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
                Answer: ## Step1: Shawn started with 5 toys. ## Step2: If he got 2 toys each from his mom and dad, then that is 4 more toys. ## Step3: 5 + 4 = 9. The answer is 9.
                
                
                Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
                Answer: ## Step1: There were originally 9 computers. ## Step2: For each of 4 days, 5 more computers were added. ## Step3: So 5 * 4 = 20 computers were added. ## Step4: 9 + 20 is 29. The answer is 29.
                
                
                Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
                Answer: ## Step1: Michael started with 58 golf balls. ## Step2: After losing 23 on tuesday, he had 58 - 23 = 35. ## Step3: After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
                
                
                Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
                Answer: ## Step1: Olivia had 23 dollars. ## Step2: 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. ## Step3: So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
            """
        )
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_tokens = 512
        self.n = 4
        self.lambda_weight = 0.5
        self.custom_chat_template = None