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
            Q: Define p = ∑{k=1}^∞ 1/k² and q = ∑{k=1}^∞ 1/k³. Find a way to write ∑{j=1}^∞ ∑{k=1}^∞ 1/(j + k)³ in terms of p and q. 
            A: ## Step1: Consider the double sum ∑{j=1}^∞ ∑{k=1}^∞ 1/(j + k)³. For a fixed n = j + k, count the number of pairs (j, k) such that j + k = n. ## Step2: The pairs are (j, k) = (1, n-1), (2, n-2), …, (n-1, 1), giving n-1 pairs, each contributing 1/n³ to the sum. ## Step3: Thus, the double sum can be rewritten as ∑{n=1}^∞ (n-1) · 1/n³ = ∑{n=1}^∞ (n-1)/n³. ## Step4: Simplify the term: (n-1)/n³ = n/n³ - 1/n³ = 1/n² - 1/n³. ## Step5: The sum becomes ∑{n=1}^∞ (1/n² - 1/n³) = ∑{n=1}^∞ 1/n² - ∑_{n=1}^∞ 1/n³ = p - q. The answer is \\boxed{p - q}.
            
            Example 2:
            Q: If f(x) = (3x-2)/(x-2), what is the value of f(-2) + f(-1) + f(0)? Express your answer as a common fraction. 
            A: ## Step1: Compute f(-2) = (3(-2)-2)/(-2-2) = (-6-2)/-4 = -8/-4 = 2. ## Step2: Compute f(-1) = (3(-1)-2)/(-1-2) = (-3-2)/-3 = -5/-3 = 5/3. ## Step3: Compute f(0) = (3(0)-2)/(0-2) = -2/-2 = 1. ## Step4: Sum the values: f(-2) + f(-1) + f(0) = 2 + 5/3 + 1. ## Step5: Convert to a common denominator: 2 = 6/3, 1 = 3/3, so 2 + 5/3 + 1 = 6/3 + 5/3 + 3/3 = 14/3. The answer is \\boxed{\\frac{14}{3}}.
            
            Example 3:
            Q: How many positive whole-number divisors does 196 have?
            A: ## Step1: Find the prime factorization of 196: 196 = 2² · 7². ## Step2: For a number n = p₁^{e₁} · p₂^{e₂}, the number of divisors is (e₁ + 1)(e₂ + 1). Here, e₁ = 2 (for prime 2) and e₂ = 2 (for prime 7). ## Step3: Calculate the number of divisors: (2+1)(2+1) = 3 · 3 = 9. ## Step4: Verify that the divisors (from combinations of 2⁰, 2¹, 2² and 7⁰, 7¹, 7²) are distinct integers: 1, 2, 4, 7, 14, 28, 49, 98, 196. The answer is \\boxed{9}.
            
            Example 4:
            Q: What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?
            A: ## Step1: The sum of three consecutive integers is (k-1) + k + (k+1) = 3k, which is a multiple of 3. ## Step2: A number n is the sum of three consecutive integers if and only if it is divisible by 3. ## Step3: A perfect cube is of the form m³. The smallest positive perfect cube divisible by 3 is 3³ = 27. ## Step4: Verify: 27 ÷ 3 = 9, so the integers are 8, 9, 10, and 8 + 9 + 10 = 27. The next cube, 4³ = 64, is not divisible by 3. The answer is \\boxed{27}.
                
            
            Example 5:
            Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
            A: ## Step1: Shawn started with 5 toys. ## Step2: If he got 2 toys each from his mom and dad, then that is 4 more toys. ## Step3: 5 + 4 = 9. The answer is \\boxed{9}.
            
            Example 6:
            Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
            A: ## Step1: There were originally 9 computers. ## Step2: For each of 4 days, 5 more computers were added. ## Step3: So 5 * 4 = 20 computers were added. ## Step4: 9 + 20 is 29. The answer is \\boxed{29}.
            
            Example 7:
            Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
            A: ## Step1: Michael started with 58 golf balls. ## Step2: After losing 23 on tuesday, he had 58 - 23 = 35. ## Step3: After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is \\boxed{33}.
            
            Example 8:
            Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
            A: ## Step1: Olivia had 23 dollars. ## Step2: 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. ## Step3: So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is \\boxed{8}.

            Now, solve the following problem step by step and end with the numeric answer in the same format:

            
            """
        )
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_tokens = 512
        self.n = 4
        self.lambda_weight = 0.5
        self.custom_chat_template = None