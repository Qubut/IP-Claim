import dspy


class CypherFromText(dspy.Signature):
    """Instructions:
    Create a Cypher MERGE statement to model all entities and relationships found in the text following these guidelines:
    - Refer to the provided schema and use existing or similar nodes, properties or relationships before creating new ones.
    - Use generic categories for node and relationship labels.
    """  # noqa: D205

    text = dspy.InputField(desc='Text to model using nodes, properties and relationships.')
    neo4j_schema = dspy.InputField(
        desc='Current graph schema in Neo4j as a list of NODES and RELATIONSHIPS.'
    )
    statement = dspy.OutputField(
        desc='Cypher statement to merge nodes and relationships found in the text.'
    )
