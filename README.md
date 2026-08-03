# Prompt Looping

> LLM - level loop -> the user holds the agent over iterative prompts, the user holds the autonomy over the responses generated throughout the course of the loop.

- You repeatedly call the model with updated prompts
- The loop is driven by **prompt → response → new prompt**
- Usually controlled manually or via a framework

### Key characteristics:

- Can be stateful/stateless, depends on if the user defines memory
- Scripting developed by the user that run iteratively
- Autonomy remains with the author of the prompts (not the llm itself)
- Static loop mechanism

## Prompt control loop

Deterministic loop controlling the llm's response; the response of the llm depends on its previous response

>Constantly evolving prompts with repeated llm calls 

```python
state = initial_input()

while not done:
	prompt = build_prompt(state)
	output = llm(prompt)
	state = update(state, output)
```

#### Without memory

- Stateless across iterations - Reponses generated as non-persistent or cannot be used in future as prompt to llm 
- Context -> Only what is passed as input by the user (Explicitly)
- Deterministic and predictable
- State managed by the user

```python
state = {prev response}
```

#### With Memory

- Memory introduced as abstraction for response persistence
- Memory externalizes state management from your loop

```python
LangChain manages:
    memory.store(messages)
    memory.inject(prompt)
```

### Iterative prompt loop

> Linear refinement -> Each iteration enhances the response by transforming/critiqueing previous response 

> Generate -> Critique -> Improve -> Simplify (Linear Flow)

- Deterministic/ Static loop
- Fixed number of steps / Converging responses 

Cons > Context growth may be non-linear 

### Recursive prompt loop

> Loop calls itself with modified inputs

```python
f(x):
	if base_case(x): return result
	sub_result = [f(x1), f(x2), f(x3) ...]
	return combine(sub_results)
```
- Non-linear structure of outputs

Cons > Context overflow
