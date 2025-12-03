I think I need to change these into scenarios

Dammit I was hoping I would have more time this is one fucking crazy disaster yet again I've had time since approximately 12:10 am its now 4:40 am 

1. Read the Persona paper 
2. Understand what data schema I need to change this into 
3. Figure out how we are going to extract the feature we are looking for 
Get a local model downloaded 
4. 

Things to think about are these 


Persona paper pipeline:
- Trait:
- Generates opposing system prompts 5 designed to elicit the trait, 5 designed to suppress it 
- Generates 40 evaluation questions to elicit trait-relevant behavior 
- Split between extraction set and evaluation set (doesnt say the split)
- Evaluation prompt is made to instruct LLM-as-a-judge to evaluate trait expression score from 0 and 100 
- Validate agreement between human evaluators & LLM judge 
- Compare questions against established benchmarks

To compute the persona vector you take the difference between the mean activations between responses that exhibit the trait & not 

1 candidate vector per layer, most informative layer is selected

Then steering 

Activation monitoring 

We'll focus on authority, self preservation

What I should do: 
- Change the data to ask the model questions

Generate 5 contrasting system prompts on trait 

40 total questions 
20 for extraction/evaluation
10 rollouts per question
5 x 20 x 10 = 1000



Confused paragraph:
To construct sequences of system prompts, we use Claude 4.0 Sonnet to generate eight prompts
that smoothly interpolate between trait-suppressing and trait-promoting instructions.4 For manyshot prompting, we use a set of 0, 5, 10, 15, or 20 examples that demonstrate the target trait. In
both settings, we generate 10 rollouts per configuration and evaluation question, and then compute
the average trait expression score over these 10 responses. We also measure the projection of the
activation at the final prompt token (the token immediately prior to the Assistant’s response) onto
the corresponding persona direction.


We need to then also create a dataset of self-preservation/authority 
