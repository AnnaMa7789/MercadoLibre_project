1. **Problem Understanding**: Brief summary of the business problem
In the context of e-commerce platform, product listing quality significantly impacts conversion rates and seller success. My task is to build a listing quality evaluation framework, based on listing and sales information for 100,000 products and to generate recommendation for those products identified as low listing quality.
2. **Methodology**:
Among 48 columns in the dataset, there are two main quantative metrics: price and sales (sold_quantity). As the product category varies from car to office supplies, I did not compare abosulte value, but instead using the price/sales ranking within each category to quantify each product's sales performance. Only category with more than 10 products are considered (~74%). Sales is the major output for quality scoring, but price is also relied upon to determine priority of fixing (for instance, if adding video link can boost 10% sales, it is recommended to work on high-value products first).

The remaining columns are qualitative metrics which provide detailed information of the product. Among them I picked the following key factors to build scoring systems. I have analyzed a few other metrics such as shipping info, picture quality, which did not make to the list due to minimal differentiation
Title Length (characters)
Title Quality Score (depending on info density, key elements, placeholder words, repeated words,etc)
Whether it has video
Whether it has been updated
Number of Pictures
Attributes Entry Count
Attributes Completeness

The weight of each factor is determined by correlation efficienct 
   - How you approached the analysis
   - Your scoring system design (dimensions, weights, rationale)
   - Segmentation strategy
4. **Key Findings**:
   - Main insights from EDA
   - Distribution of quality scores
   - Patterns identified
5. **Recommendations**:
   - Top improvement opportunities
   - Prioritization approach
   - Estimated business impact
6. **GenAI Usage**:
   - Tools used and why
   - Example prompts and results
   - Value delivered

## The dataset has 100,000 product listing entries, with 48 columns. The main quantative metrics can be grouped into two main aspects: price (base_price, original_price,etc ) and sales (sold_quantity, available_quantity). 
sales:
The total sales are 239,699, which indicates a relatively low (2.39) sales per product. These sales concentrated on less than 25% of products.
prices: median price is 250, variation range is very large due to differents product types, from car to usb driver. 99.99% of product do not chang price

The remaining columns are qualitative metrics which describle and provide detailed listing of the product. Among them I picked the following key features: 
titles: there are 98,823 unique titles grouped under 10,907 categories. Average title length is 45 characters (ranging from 1 to 100). length of title has positive correlation with sales with correlation coefficient = 0.625
video: only 2985 (~3%) products with video links. The presence of video can help sales boost within category
image: about 33.3% products has 1 picture, about 64.3% has 2-6 pictures. only 789 product (~ 0.8%) missing pictures. majority of picture size is 500*375 or 500*500 and majority of max pricture size 1200*900:
update frequency: ~70% of listing never update since its creation, ~30% are updated within 2 months. upddate frequency does not seem to have significant impact on sales
attribute completeness: ~87% products has blank attribute and ~10% have 1 or 2 entries. within those products who have at least one entries, about 80% has complete field information.  entry number (correlation  0.039) and completeness (correlation 0.091 ) both have positive impact on sales, but not significant. 
shipping info completeness：all products have shipping and there is mininal differentiation regarding completeness. 


common quality issue: short title& low quality title, missing video link, lack of updating, picutre? attribute entry and completeness36
