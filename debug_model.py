
from dotenv import load_dotenv
load_dotenv()

from chains import qa_chain

context = """
[Page 6]
Renewable resources are naturally replenished over time.
The renewable resources discussed in this section are solar energy,
wind energy, water, biomass, and geothermal energy.
"""

result = qa_chain.invoke({
    "context": context,
    "question": "List some renewable resources."
})

print(result)
