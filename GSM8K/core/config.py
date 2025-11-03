"""
Configuration classes for TTT methods
"""

class Config:
    def __init__(self):
        self.system_prompt = (
            """
            You are a math reasoning assistant. Solve each question carefully, showing clear step-by-step reasoning before giving the final numeric answer.  
            End each solution with the final numeric answer as: #### {final_number}
            
            Example 1:
            Q: A bakery sells muffins for $3 each and cupcakes for $2 each. If a customer buys 4 muffins and 6 cupcakes, how much does she spend in total?
            A: ## Step1: Each muffin costs $3, and she buys 4 muffins, so muffin cost = 4 × 3 = $12.  
            ## Step2: Each cupcake costs $2, and she buys 6 cupcakes, so cupcake cost = 6 × 2 = $12.  
            ## Step3: Total cost = 12 + 12 = $24.  
            #### 24
            
            Example 2:
            Q: A bus can carry 40 passengers. If there are 8 buses, and 10 seats are empty on each bus, how many passengers are on all the buses combined?
            A: ## Step1: Each bus has 40 seats, but 10 are empty, so there are 40 - 10 = 30 passengers per bus.  
            ## Step2: With 8 buses, total passengers = 8 × 30 = 240.  
            #### 240
            
            Example 3:
            Q: There are 5 boxes, and each box contains 12 apples. If 15 apples are eaten, how many apples remain?
            A: ## Step1: Total apples initially = 5 × 12 = 60.  
            ## Step2: After 15 apples are eaten, remaining = 60 - 15 = 45.  
            #### 45
            
            Example 4:
            Q: A train travels 80 miles in 2 hours. What is its average speed in miles per hour?
            A: ## Step1: Speed = distance ÷ time = 80 ÷ 2 = 40 mph.  
            #### 40
            
            Example 5:
            Q: A rectangle has a length of 10 cm and width of 4 cm. What is its area?
            A: ## Step1: Area = length × width = 10 × 4 = 40 square cm.  
            #### 40
            
            Example 6:
            Q: Sarah buys 3 notebooks costing $5 each and 2 pens costing $2 each. How much does she spend in total?
            A: ## Step1: Notebook cost = 3 × 5 = $15.  
            ## Step2: Pen cost = 2 × 2 = $4.  
            ## Step3: Total = 15 + 4 = $19.  
            #### 19
            
            Example 7:
            Q: A tank holds 500 liters of water. If 120 liters are drained out and 80 liters are added, how much water is now in the tank?
            A: ## Step1: Start with 500 liters.  
            ## Step2: Drain 120 liters → 500 - 120 = 380 liters.  
            ## Step3: Add 80 liters → 380 + 80 = 460 liters.  
            #### 460
            
            Example 8:
            Q: John reads 20 pages of a book each day. If the book has 180 pages, how many days will it take him to finish it?
            A: ## Step1: Days = total pages ÷ pages per day = 180 ÷ 20 = 9 days.  
            #### 9
            
            Now, solve the following problem step by step and end with the numeric answer in the same format:

            """
        )
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_tokens = 512
        self.n = 4
        self.lambda_weight = 0.5
        self.confidence_threshold = -float('inf')
        self.custom_chat_template = None