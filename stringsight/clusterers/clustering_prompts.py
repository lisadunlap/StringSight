clustering_systems_prompt = f"""You are an expert machine learning engineer tasked with summarizing LLM response behaviors. Given a list of properties seen in LLM responses that belong to the same cluster, create a clear description (1-2 short sentences) that accurately describes most or all properties in the cluster. This should be a specific behavior of a model response, not a category of behaviors. Think: if a user saw this property, would they be able to understand the model behavior and gain valuable insight about the models specific behavior on a task?

For instance "Speaking Tone and Emoji Usage" is a category, but "uses an enthusiastic tone" is a specific behavior. Descriptions like "Provides detailed math responses" are not informative because they could apply to many clusters. Instead, describe the behavior in a way that is specific and informative to this cluster, even if it doesn't apply to all properties.

Avoid filler words like "detailed", "comprehensive" or "step-by-step" unless explicitly mentioned in the properties. Focus on the main behavior of the cluster.

Consider whether a user could easily understand the model's behavior and come up with an example scenario described by this behavior. If given a model response, could a user determine whether the model is exhibiting this behavior?

Output the cluster behavior description and nothing else."""

deduplication_clustering_systems_prompt = """You are a machine learning expert evaluating LLM output behaviors. Given a list of behaviors seen in LLM outputs across a dataset, merge those that are redundant or very similar, keeping the most informative and specific version. Think about if a user would gain any new information from seeing both behaviors.

Each behavior should be 1-2 clear and concise sentences. Avoid vague, broad, or meta-properties—focus on specific behaviors. Only use terms like "detailed", "comprehensive", or "step-by-step" if they are central to the behavior. Refrain from having high word overlap between your final properties, as this typically indicates that these are filler words (e.g. "clear", "comprehensive", "consistenly" etc). Again, your final list should not have multiple properties that start with the same few words. 

If two behaviors in the list are opposites (e.g., "uses X" and "doesn't use X"), keep both. Do not combine many behaviors into one summary or talk about the variation of behaviors, each behavior should be a single property that someone can easily identify if looking at a model response. Each behavior should be 1-2 sentences.

Think: if a user saw this property, would they be able to understand the model behavior and gain valuable insight about the models specific behavior on a task?

Output a plain list: one behavior per line, no numbering or bullets.
"""

outlier_clustering_systems_prompt = """You are a machine learning expert specializing in the behavior of large language models. 

I will provide you with a list of fine-grained behaviors of an LLM on a task. Your task is to cluster the behaviors into groups that are similar. Each group should be a single behavior that is representative of the group. Note that some behaviors may not belong to any group, which is fine, we are just trying to find the most interesting and informative behaviors that appear at least 5 times in the data.

Instructions:
1. Analyze all the fine-grained behaviors
2. Cluster the behaviors into at most {max_coarse_clusters}. Each group should be a single behavior that is representative of the group. Ensure that the behaviors in a cluster are not opposites of each other (e.g., "uses X" and "doesn't use X"), these should be in separate clusters. Do not combine many behaviors into one summary or talk about the variation of behaviors, each behavior should be a single property that someone can easily identify if looking at a model response.
3. Create clear, descriptive names for each cluster. Each cluster name should be 1-2 sentences decribing the behavior.  
4. Output ONLY the cluster names, one per line. Do not include numbering, bullets, or other formatting - just the plain cluster names
"""

coarse_clustering_systems_prompt = """You are a machine learning expert specializing in the behavior of large language models. 

I will provide you with a list of fine-grained properties describing model behavior. Your task is to create {max_coarse_clusters} broader property names that capture the high-level themes across these properties.

Instructions:
1. Analyze all the fine-grained properties
2. Identify {max_coarse_clusters} major properties
3. Create clear, descriptive names for each property
4. Each property should be 1-2 sentences that capture the essence of that property
5. Output ONLY the property names, one per line
6. Do NOT include numbering, bullets, or other formatting - just the plain property names

Focus on creating properties that are:
- Distinct from each other
- Broad enough to encompass multiple fine-grained properties
- Descriptive and meaningful for understanding model behavior"""