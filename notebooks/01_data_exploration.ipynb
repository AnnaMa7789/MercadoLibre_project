############################ EDA ########################################
import duckdb
import pandas as pd
import numpy as np
import json
import data_loader as dl
import  quality_scorer as qs

## load data into df
listings = dl.load_data()
print(f"Loaded {len(listings):,} listings")

df = pd.DataFrame(listings)
print(f"data shape: {df.shape}")
print(f"column names: {df.columns.tolist()}")

## create duckdb LINK
conn = duckdb.connect()
conn.register('listings', df)


## sales distribution
basic_stats = conn.execute("""
SELECT 
    -- Basic statistics
    MAX(sold_quantity) as max_sold,
    MIN(sold_quantity) as min_sold,
    AVG(sold_quantity) as avg_sold,
    sum(sold_quantity) as total_sold,
    
    -- Median (50th percentile)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_quantity) as median_sold,
    
    -- 25th and 75th percentiles (Q1 and Q3)
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY sold_quantity) as p10_sold,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sold_quantity) as p25_sold,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sold_quantity) as p75_sold,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY sold_quantity) as p90_sold,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY sold_quantity) as p95_sold,
    
    -- Count statistics
    COUNT(*) as total_listings,
    COUNT(sold_quantity) as non_null_sold,

FROM listings
WHERE sold_quantity IS NOT NULL;
""").fetchdf()
print(basic_stats)

## price distribution
basic_stats = conn.execute("""
    SELECT 
         price/ base_price as price_change, count(1) FROM listings
        group by 1
""").fetchdf()
print(basic_stats)

basic_stats = conn.execute("""
    SELECT 
    MAX(price) as max_price,
    MIN(price) as min_price,
    AVG(price) as avg_price,
   -- Median (50th percentile)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
    
    -- 25th and 75th percentiles (Q1 and Q3)
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY price) as p10_price,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25_price,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75_price,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY price) as p90_price,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price) as p95_price,	  FROM listings
        where price

""").fetchdf()
print(basic_stats)

## title analysis
## get distribution
basic_stats = conn.execute("""
    select count(distinct title) as distinct_title,
    MAX(cnt) as max_cnt,
    MIN(cnt) as min_cnt,
    AVG(cnt) as avg_cnt,
    MAX(length) as max_length,
    MIN(length) as min_length,
    AVG(length) as avg_length,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY length) as p10_length,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY length) as p25_length,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY length) as p75_length,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY length) as p90_length,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY length) as p95_length,
    from (SELECT 
        title,len(title) as length, count(1) as cnt  FROM listings
        group by 1)
""").fetchdf()
print(basic_stats)

## title length and sales correlation
basic_stats = conn.execute("""
    with ranked_data as  (SELECT 
    title,
    category_id,
    length,
    sold_quantity,
    -- Length ranking within category
    RANK() OVER (PARTITION BY category_id ORDER BY length DESC) as length_rank_in_category,
    -- Sales ranking within category
    RANK() OVER (PARTITION BY category_id ORDER BY sold_quantity DESC) as sales_rank_in_category
FROM (
    SELECT 
        title,
        category_id,
        LENGTH(title) as length,
        SUM(sold_quantity) as sold_quantity,
        -- Count titles per category for filtering
        COUNT(DISTINCT title) OVER (PARTITION BY category_id) as titles_per_category
    FROM listings
    WHERE title IS NOT NULL 
      AND category_id IS NOT NULL
      and sold_quantity>0
    GROUP BY title, category_id
) as title_stats
WHERE titles_per_category >= 10  -- Keep only categories with 10+ distinct titles

)
SELECT 
    -- Check correlation
    CORR(length_rank_in_category, sales_rank_in_category) as correlation_coefficient,
    -- Compare top rankings
    COUNT(CASE WHEN length_rank_in_category <= 10 AND sales_rank_in_category <= 10 THEN 1 END) as both_top_10,
    COUNT(CASE WHEN length_rank_in_category > 10 AND sales_rank_in_category > 10 THEN 1 END) as both_bottom,
    COUNT(CASE WHEN length_rank_in_category <= 10 AND sales_rank_in_category > 10 THEN 1 END) as long_title_low_sales,
    COUNT(CASE WHEN length_rank_in_category > 10 AND sales_rank_in_category <= 10 THEN 1 END) as short_title_high_sales
FROM ranked_data;
""").fetchdf()
print(basic_stats)

## get distribution of category
basic_stats = conn.execute("""
    SELECT 
        count(distinct category_id)  FROM listings

""").fetchdf()
print(basic_stats)

## title quality analysis
# Run the analysis
df, filtered_df, correlation_results, performance_df = qs.analyze_title_quality_distribution_and_correlation(df)

# Display key insights
print("\n" + "="*70)
print("🔑 KEY INSIGHTS FROM TITLE QUALITY ANALYSIS")
print("="*70)

# Get the correlation with title_score
title_score_corr = next(r for r in correlation_results if r['metric'] == 'title_score')
print(f"\n1. Overall Title Quality Correlation:")
print(f"   • Correlation with sales rank: {title_score_corr['corr_with_sales_rank']:.3f}")
print(f"     (Negative = better titles have better sales ranking)")
print(f"   • Correlation with sold quantity: {title_score_corr['corr_with_sold_quantity']:.3f}")
print(f"     (Positive = better titles sell more units)")

# Most impactful metric
sorted_correlations = sorted(correlation_results, key=lambda x: abs(x['corr_with_sales_rank']), reverse=True)
top_metric = sorted_correlations[0]
print(f"\n2. Most Impactful Metric:")
print(f"   • {top_metric['metric']}: correlation = {top_metric['corr_with_sales_rank']:.3f}")
print(f"     (Strongest relationship with sales ranking)")

# Quality category performance
if len(performance_df) >= 2:
    best_category = performance_df.iloc[0]  # Lowest percentile rank
    worst_category = performance_df.iloc[-1]  # Highest percentile rank
    
    print(f"\n3. Performance Difference:")
    print(f"   • {best_category['title_quality_category']}: avg rank = {best_category['avg_percentile_rank']:.1f}%")
    print(f"   • {worst_category['title_quality_category']}: avg rank = {worst_category['avg_percentile_rank']:.1f}%")
    print(f"   • Difference: {worst_category['avg_percentile_rank'] - best_category['avg_percentile_rank']:.1f}% points")

## video analysis
## get distribution 
basic_stats = conn.execute("""
    SELECT 
        case when video_id is not null then 1 else 0 end as with_video, count(1)  FROM listings
        group by 1

""").fetchdf()
print(basic_stats)

## video and sales correlation
basic_stats = conn.execute("""
    WITH video_impact AS (
    SELECT 
        case when video_id is not null then 1 else 0 end as with_video,
        category_id,
        RANK() OVER (PARTITION BY category_id ORDER BY SUM(sold_quantity) DESC) as category_sales_rank,
        COUNT(*) OVER (PARTITION BY category_id) as titles_in_category
    FROM listings
    WHERE title IS NOT NULL 
      AND category_id IS NOT NULL
      AND sold_quantity >0
    GROUP BY title, category_id, video_id
)
SELECT 
    with_video,
    COUNT(*) as total_listings,
    ROUND(AVG(category_sales_rank), 1) as average_rank,
    ROUND(AVG(category_sales_rank * 100.0 / titles_in_category), 1) as average_percentile,
    -- Percentage in top ranks
    ROUND(SUM(CASE WHEN category_sales_rank <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_top_10,
    ROUND(SUM(CASE WHEN category_sales_rank <= titles_in_category * 0.25 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_top_25_percent
FROM video_impact
where titles_in_category >=10
GROUP BY with_video
ORDER BY average_percentile;
""").fetchdf()
print(basic_stats)

## update frequency analysis
## get distribution 
basic_stats = conn.execute("""
    select case when date_update_gap=0 then 'never update'
        when date_update_gap between 1 and 60 then 'update within two months'
        when date_update_gap>61  then 'update after two months'
        else null end as date_update_gap_group,sum(cnt) from  (SELECT 
        cast(last_updated as date) - cast(date_created as date) as date_update_gap, 
        
         count(1) as cnt  FROM listings
        group by 1
        order by 1)
        group by 1

""").fetchdf()
print(basic_stats)

## update frequency and sales relationship
basic_stats = conn.execute("""
    WITH listing_ranks AS (
    SELECT 
        id,
        title,
        category_id,
        sold_quantity,
        date_created,
        last_updated,
        -- Calculate update gap in days
        cast(last_updated as date) - cast(date_created as date) as update_gap_days,
        -- Flag for never updated
        CASE 
            WHEN cast(last_updated as date) - cast(date_created as date) =0 THEN 'Never Updated'
            ELSE 'Updated'
        END as update_status,
        -- Sales rank within category
        RANK() OVER (PARTITION BY category_id ORDER BY sold_quantity DESC) as sales_rank_in_category,
        -- Count in category for percentile
        COUNT(*) OVER (PARTITION BY category_id) as listings_in_category
    FROM listings
    WHERE category_id IS NOT NULL
      AND sold_quantity >0 
      AND date_created IS NOT NULL
      AND last_updated IS NOT NULL
),
rank_comparison AS (
    SELECT 
        update_status,
        COUNT(*) as listing_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
        -- Sales performance metrics
        AVG(sales_rank_in_category) as avg_sales_rank,
        MEDIAN(sales_rank_in_category) as median_sales_rank,
        -- Percentile rank (lower is better)
        AVG(sales_rank_in_category * 100.0 / listings_in_category) as avg_percentile_rank,
        -- Top performers
        SUM(CASE WHEN sales_rank_in_category <= 10 THEN 1 ELSE 0 END) as top_10_count,
        SUM(CASE WHEN sales_rank_in_category <= listings_in_category * 0.25 THEN 1 ELSE 0 END) as top_25_percent_count
    FROM listing_ranks
    where listings_in_category>=10
    GROUP BY update_status
)
SELECT * FROM rank_comparison
ORDER BY avg_sales_rank;

""").fetchdf()
print(basic_stats)

## picture analysis
counts, sizes, max_sizes = qs.safe_get_picture_info(df)

df['picture_count'] = counts
df['first_size'] = sizes
df['first_max_size'] = max_sizes

print("✅ done!")

print("Picture count distribution:")
pic_count_dist = df['picture_count'].value_counts().sort_index()
for count, freq in pic_count_dist.items():
    print(f"  {count} pictures: {freq} products ({(freq/len(df)*100):.1f}%)")

print("Picture size distribution:")
if df['first_size'].notna().any():
    size_dist = df['first_size'].value_counts().head(10)  # top 10 most common sizes
    for size, count in size_dist.items():
        print(f"  size {size}: {count} products ({(count/len(df)*100):.1f}%)")
else:
    print("no info")

print("Max Size distribution:")
if df['first_max_size'].notna().any():
    size_dist = df['first_max_size'].value_counts().head(10)  # top 10 most common sizes
    for size, count in size_dist.items():
        print(f"  size {size}: {count} products ({(count/len(df)*100):.1f}%)")
else:
    print("no info")



## attribute analysis
entry_counts, empty_counts, total_counts = qs.safe_analyze_attributes(df)
# Add results to DataFrame
df['attr_entries'] = entry_counts
df['attr_empty_fields'] = empty_counts
df['attr_total_fields'] = total_counts

df['attr_completeness_pct'] = df.apply(qs.calculate_completeness, axis=1)

print(f"\n✅ Successfully added columns:")
print(f"   - attr_entries: Number of attribute entries")
print(f"   - attr_empty_fields: Number of empty fields")
print(f"   - attr_total_fields: Total number of fields")
print(f"   - attr_completeness_pct: Field completeness percentage")

# Show basic statistics with distribution information
print(f"\n📊 BASIC STATISTICS:")
print(f"Total rows analyzed: {len(df):,}")

# 1. Distribution of entry counts - show each unique value
print(f"\n1. ATTRIBUTE ENTRY COUNT DISTRIBUTION:")
print("Each entry count value and how many IDs have that value:")

# Get value counts for entry counts
entry_value_counts = df['attr_entries'].value_counts().sort_index()

# Show all unique values
for entry_count, id_count in entry_value_counts.items():
    percentage = (id_count / len(df)) * 100
    print(f"  {entry_count} entries: {id_count:,} IDs ({percentage:.1f}%)")

print(f"\nSummary:")
print(f"  Unique entry count values: {len(entry_value_counts)}")
print(f"  Most common: {entry_value_counts.index[0]} entries ({entry_value_counts.iloc[0]:,} IDs)")
print(f"  IDs with 0 entries: {entry_value_counts.get(0, 0):,} IDs")
print(f"  IDs with ≥1 entries: {len(df) - entry_value_counts.get(0, 0):,} IDs")


# Add formatted missing fields column
filtered_df['missing_fields_format'] = filtered_df.apply(qs.format_missing_fields, axis=1)

# Group by the formatted missing fields and count IDs
missing_distribution = filtered_df['missing_fields_format'].value_counts()

print(f"\nDistribution of missing fields (sorted by ID count, descending):")
print("-" * 70)

# Show all groups
rank = 1
for missing_format, id_count in missing_distribution.items():
    percentage = (id_count / len(filtered_df)) * 100
    print(f"{rank:2d}. {missing_format:40s}: {id_count:6,} IDs ({percentage:5.1f}%)")
    rank += 1

# Run the analysis
filtered_df, results_df = qs.analyze_attribute_impact_on_sales(df)

# Display results
# Using pandas DataFrame display with formatting
display_df = results_df.copy().sort_values('avg_percentile_rank')

# Format columns
display_df['product_count'] = display_df['product_count'].apply(lambda x: f"{x:,}")
display_df['avg_sales_rank'] = display_df['avg_sales_rank'].apply(lambda x: f"{x:.1f}")
display_df['avg_percentile_rank'] = display_df['avg_percentile_rank'].apply(lambda x: f"{x:.1f}")
display_df['avg_sold_quantity'] = display_df['avg_sold_quantity'].apply(lambda x: f"{x:.1f}")
display_df['pct_top_10'] = display_df['pct_top_10'].apply(lambda x: f"{x:.1f}%")
display_df['pct_top_25_percent'] = display_df['pct_top_25_percent'].apply(lambda x: f"{x:.1f}%")
display_df['percentage_of_total'] = display_df['percentage_of_total'].apply(lambda x: f"{x:.2f}%")

# Rename columns for display
display_df = display_df.rename(columns={
    'entry_group': 'Entries',
    'completeness_group': 'Completeness',
    'product_count': 'Products',
    'avg_sales_rank': 'Avg Rank',
    'avg_percentile_rank': 'Avg %Rank',
    'avg_sold_quantity': 'Avg Sold',
    'pct_top_10': '% Top 10',
    'pct_top_25_percent': '% Top 25%',
    'percentage_of_total': '% of Total'
})

print(f"\n📊 ATTRIBUTE QUALITY IMPACT ON SALES PERFORMANCE")
print("="*120)
print(display_df.to_string(index=False))



# Statistical correlation analysis
print(f"\n📈 CORRELATION ANALYSIS")
print("="*70)

# Calculate correlations
correlations = filtered_df[['attr_entries', 'attr_completeness_pct', 'sales_rank']].corr()

print("Correlation matrix (Pearson):")
print("(Note: Negative correlation with sales_rank/percentile_rank means better sales)")
print(correlations.round(3))

# 2. Distribution of completeness percentages - only for IDs with 1 or 2 entries
print(f"\n2. COMPLETENESS DISTRIBUTION (ONLY 1 OR 2 ENTRIES):")

# Filter for IDs with 1 or 2 entries
filtered_df = df[df['attr_entries'].isin([1, 2])].copy()

print(f"Analyzing {len(filtered_df):,} IDs with 1 or 2 attribute entries")
print(f"  IDs with 1 entry: {(filtered_df['attr_entries'] == 1).sum():,}")
print(f"  IDs with 2 entries: {(filtered_df['attr_entries'] == 2).sum():,}")

## shipping analysis
print("🔍 Analyzing shipping field...")

# Initialize lists for each field
has_shipping_data = []
local_pick_up_vals = []
free_shipping_vals = []
mode_vals = []
dimensions_vals = []
methods_counts = []
tags_counts = []

# Process each shipping value
for i, shipping_value in enumerate(df['shipping']):
    if i % 10000 == 0 and i > 0:
        print(f"  Processed {i:,} rows...")
    
    analysis = qs.analyze_shipping_field(shipping_value)
    
    has_shipping_data.append(analysis['has_shipping_data'])
    local_pick_up_vals.append(analysis['local_pick_up'])
    free_shipping_vals.append(analysis['free_shipping'])
    mode_vals.append(analysis['mode'])
    dimensions_vals.append(analysis['dimensions'])
    methods_counts.append(analysis['methods_count'])
    tags_counts.append(analysis['tags_count'])

# Add to DataFrame
df['shipping_has_data'] = has_shipping_data
df['shipping_local_pick_up'] = local_pick_up_vals
df['shipping_free_shipping'] = free_shipping_vals
df['shipping_mode'] = mode_vals
df['shipping_dimensions'] = dimensions_vals
df['shipping_methods_count'] = methods_counts
df['shipping_tags_count'] = tags_counts

print(f"\n✅ Shipping analysis complete! Processed {len(df):,} rows")

# Display distribution for each field
print(f"\n📊 SHIPPING FIELD DISTRIBUTION ANALYSIS")
print("="*70)

# 1. Has shipping data
print(f"\n1. HAS SHIPPING DATA:")
has_data_count = df['shipping_has_data'].sum()
no_data_count = len(df) - has_data_count
print(f"   With shipping data: {has_data_count:,} IDs ({(has_data_count/len(df)*100):.1f}%)")
print(f"   Without shipping data: {no_data_count:,} IDs ({(no_data_count/len(df)*100):.1f}%)")

# 2. Local pick-up distribution
print(f"\n2. LOCAL PICK-UP DISTRIBUTION:")
if df['shipping_local_pick_up'].notna().any():
    local_pickup_dist = df['shipping_local_pick_up'].value_counts(dropna=False)
    for value, count in local_pickup_dist.items():
        percentage = (count / len(df)) * 100
        value_str = "True" if value is True else "False" if value is False else "None"
        print(f"   {value_str}: {count:,} IDs ({percentage:.1f}%)")
else:
    print("   No local_pick_up data")

# 3. Free shipping distribution
print(f"\n3. FREE SHIPPING DISTRIBUTION:")
if df['shipping_free_shipping'].notna().any():
    free_shipping_dist = df['shipping_free_shipping'].value_counts(dropna=False)
    for value, count in free_shipping_dist.items():
        percentage = (count / len(df)) * 100
        value_str = "True" if value is True else "False" if value is False else "None"
        print(f"   {value_str}: {count:,} IDs ({percentage:.1f}%)")
else:
    print("   No free_shipping data")

# 4. Mode distribution
print(f"\n4. SHIPPING MODE DISTRIBUTION:")
if df['shipping_mode'].notna().any():
    mode_dist = df['shipping_mode'].value_counts(dropna=False)
    for mode_value, count in mode_dist.items():
        percentage = (count / len(df)) * 100
        mode_str = str(mode_value) if mode_value is not None else "None"
        print(f"   '{mode_str}': {count:,} IDs ({percentage:.1f}%)")
else:
    print("   No mode data")

# 5. Dimensions distribution
print(f"\n5. DIMENSIONS DISTRIBUTION:")
if df['shipping_dimensions'].notna().any():
    dimensions_dist = df['shipping_dimensions'].value_counts(dropna=False)
    # Show only top values if too many
    if len(dimensions_dist) <= 10:
        for dim_value, count in dimensions_dist.items():
            percentage = (count / len(df)) * 100
            dim_str = str(dim_value) if dim_value is not None else "None"
            print(f"   '{dim_str}': {count:,} IDs ({percentage:.1f}%)")
    else:
        # Show top 10
        print(f"   (Showing top 10 out of {len(dimensions_dist)} unique values)")
        top_dimensions = dimensions_dist.head(10)
        for dim_value, count in top_dimensions.items():
            percentage = (count / len(df)) * 100
            dim_str = str(dim_value) if dim_value is not None else "None"
            print(f"   '{dim_str}': {count:,} IDs ({percentage:.1f}%)")
        # Show None count separately
        if None in dimensions_dist.index:
            none_count = dimensions_dist[None]
            print(f"   'None': {none_count:,} IDs ({(none_count/len(df)*100):.1f}%)")
else:
    print("   No dimensions data")

# 6. Methods count distribution
print(f"\n6. SHIPPING METHODS COUNT DISTRIBUTION:")
if df['shipping_methods_count'].notna().any():
    methods_dist = df['shipping_methods_count'].value_counts().sort_index()
    for count_value, id_count in methods_dist.items():
        percentage = (id_count / len(df)) * 100
        print(f"   {count_value} methods: {id_count:,} IDs ({percentage:.1f}%)")
    
    # Summary
    print(f"\n   Methods count summary:")
    print(f"     Average methods: {df['shipping_methods_count'].mean():.2f}")
    print(f"     Max methods: {df['shipping_methods_count'].max()}")
    print(f"     IDs with 0 methods: {methods_dist.get(0, 0):,}")
else:
    print("   No methods count data")

# 7. Tags count distribution
print(f"\n7. SHIPPING TAGS COUNT DISTRIBUTION:")
if df['shipping_tags_count'].notna().any():
    tags_dist = df['shipping_tags_count'].value_counts().sort_index()
    for count_value, id_count in tags_dist.items():
        percentage = (id_count / len(df)) * 100
        print(f"   {count_value} tags: {id_count:,} IDs ({percentage:.1f}%)")
    
    # Summary
    print(f"\n   Tags count summary:")
    print(f"     Average tags: {df['shipping_tags_count'].mean():.2f}")
    print(f"     Max tags: {df['shipping_tags_count'].max()}")
    print(f"     IDs with 0 tags: {tags_dist.get(0, 0):,}")
else:
    print("   No tags count data")



#### END OF ANALYSIS, SAVE FILE 
df.to_csv('processed_df.csv', index=False) 


