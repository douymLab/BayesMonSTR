#!/usr/bin/env Rscript

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments is provided
if (length(args) < 3) {
  stop("Usage: Rscript script.R <base_df_path> <read_df_path> <reference_df_path> [output_path] [mut_start] [mut_end] [mut_type]")
}

# Parse arguments
base_df_path <- args[1]
read_df_path <- args[2]
reference_df_path <- args[3]
output_file <- if (length(args) >= 4) args[4] else "output_default_test.png"
mut_start <- if (length(args) >= 5) as.numeric(args[5]) else "None"
mut_end <- if (length(args) >= 6) as.numeric(args[6]) else "None"
mut_type <- if (length(args) >= 7) args[7] else "None"
start_axis <- if (length(args) >= 8) as.numeric(args[8]) else 0

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(grid)
  library(gtable)
  library(tidyr)
  library(dplyr)
  library(purrr)
  library(stringr)
})

# Configuration settings ----
# Configuration settings ----
config <- list(
  file_paths = list(
    base_df = base_df_path,
    read_df = read_df_path,
    reference_df = reference_df_path
  ),
  visual = list(
    colors = list(
      base = c(
        'A' = '#33a02c',
        'T' = '#1f78b4',
        'G' = '#fccde5',
        'C' = '#fdb462',
        'N' = '#444444',
        '-' = '#ffffff'
      ),
      state = c(
        "INSERTION" = "#54278f",
        "MATCH" = "#d9d9d9",
        "DELETION" = "transparent"
      ),
      border = c(
        'MATCH' = "transparent",
        'MISMATCH' = "transparent",
        'DELETION' = "#1e1e1e",
        'REF' = "#1e1e1e",
        'HARD_CLIPPED' = "#1e1e1e",
        'SOFT_CLIPPED' = "#1e1e1e"
      )
    ),
    tiles = list(
      width = c(
        'MATCH' = 1,
        'MISMATCH' = 1,
        'DELETION' = 0.75,
        'INSERTION' = 0.75,
        'REF' = 0.9,
        'HARD_CLIPPED' = 0.82,
        'SOFT_CLIPPED' = 0.82
      ),
      height = c(
        'MATCH' = 1,
        'MISMATCH' = 1,
        'DELETION' = 0.2,
        'INSERTION' = 0.75,
        'REF' = 0.9,
        'HARD_CLIPPED' = 0.82,
        'SOFT_CLIPPED' = 0.82
      ),
      border = c(
        "MATCH" = 0,
        "MISMATCH" = 0,
        'DELETION' = 0.05,
        'HARD_CLIPPED' = 0.18,
        'SOFT_CLIPPED' = 0.18,
        'REF' = 0.15
      )
    ),
    annotation = list(
      read_point_size = 0.3,
      reverse_point_size = 0.6,
      annotate_line_width = 0.1,
      phasing_line_width = 0.03
    ),
    font = list(
      size = 1,
      color = 'black'
    )
  )
)

# Data import ----
data_import <- function(paths) {
  lapply(paths, function(p) read_csv(p, show_col_types = FALSE))
}

imported_data <- data_import(config$file_paths)
list2env(imported_data, envir = .GlobalEnv)

# Data preprocessing ----
has_intron <- any(c("INTRON_START", "INTRON_END") %in% base_df$align_state)
# Handle INTRON segments and mutation range
if(has_intron){
  base_df <- base_df %>%
    mutate(visual_x = as.numeric(visual_x)) %>%
    group_by(y) %>%
    mutate(
      segment_id = cumsum(align_state %in% c("INTRON_START", "INTRON_END")),
      segment_id = as.integer(factor(segment_id))
    ) %>%
    group_by(y, segment_id) %>%
    mutate(
      segment_start = min(visual_x),
      segment_end = max(visual_x)
    ) %>%
    # 添加全局坐标范围计算
    mutate(
      global_start = if (mut_start == "None" && mut_end == "None") min(visual_x) else NA,
      global_end = if (mut_start == "None" && mut_end == "None") max(visual_x) else NA
    ) %>%
    ungroup() %>%
    group_by(y, segment_id) %>%
    filter(
      # 当有突变范围时
      if (mut_start != "None" || mut_end != "None") {
        # 检查是否存在有效突变区间
        has_valid_mut_range <- !is.na(as.numeric(mut_start)) && !is.na(as.numeric(mut_end))
        # 生成与segment_end相同长度的逻辑向量
        in_mut_range <- rep(has_valid_mut_range, length(segment_end)) & 
          (segment_end >= as.numeric(mut_start) & segment_start <= as.numeric(mut_end))
        
        # 组合过滤条件（移除保留全部的条件）
        in_mut_range
      } 
      # 当无突变范围时
      else {
        # 保留与其他reads重叠的区域或后半部分
        (segment_end >= global_start & segment_start <= global_end) |  # 与其他reads重叠
          (segment_start > (global_start + global_end)/2) # 保留后半部分
      }
    ) %>%
    ungroup() %>%
    select(-segment_id,-segment_start, -segment_end, -global_start, -global_end) %>%
    filter(!align_state %in% c("INTRON_START", "INTRON_END"))
  
  # Update reference_start reference_end xmin	xmax of read_df
  read_df <- base_df %>%
    filter(align_state != "REF") %>%  # 排除参考序列
    group_by(read_id) %>%
    summarise(
      new_reference_start = min(ref_pos, na.rm = TRUE),
      new_reference_end = max(ref_pos, na.rm = TRUE),
      new_xmin = min(visual_x, na.rm = TRUE),
      new_xmax = max(visual_x, na.rm = TRUE)
    ) %>%
    inner_join(read_df, by = "read_id") %>%
    mutate(
      reference_start = coalesce(new_reference_start, reference_start),
      reference_end = coalesce(new_reference_end, reference_end),
      xmin = coalesce(new_xmin, xmin),
      xmax = coalesce(new_xmax, xmax)
    ) %>%
    select(-starts_with("new_"))
  
  # Update reference_df
  reference_df <- reference_df %>%
    semi_join(
      read_df %>%
        transmute(visual_x = map2(xmin, xmax, ~ seq(.x, .y + 1))) %>%
        unnest(visual_x),
      by = "visual_x"
    )
}

# Update align_state for mismatches before rbind
base_df <- base_df %>%
  left_join(reference_df %>% select(visual_x, base), by = "visual_x") %>%
  mutate(
    align_state = case_when(
      # 保持原有的特殊状态
      align_state %in% c("SOFT_CLIPPED", "DELETION", "HARD_CLIPPED") ~ align_state,
      # 检查base是否与参考序列不同
      base.x != base.y ~ "MISMATCH",
      # 其他情况保持原状态
      TRUE ~ align_state
    )
  ) %>%
  select(-base.y) %>%  # 移除参考序列的base列
  rename(base = base.x)  # 恢复原有的base列名

# Data processing ----
base_df <- bind_rows(reference_df, base_df)
range_x <- range(reference_df$visual_x)
range_y <- range(base_df$y)
base_df$base_quality_binary <- ifelse(base_df$base_quality >= 30, '>=30', '<30')

## is reverse data
reverse_data <- read_df %>%
  transmute(
    read_id,
    y = y_position,
    ref_start = reference_start + 1,
    ref_end = reference_end - 1,
    is_reverse
  )

# 获取错配位置
mismatch_positions <- base_df %>%
  filter(align_state == "MISMATCH") %>%
  select(visual_x, y) %>%
  distinct()

# 展开序列并过滤掉错配位置
reverse_data$base_x <- Map(seq, reverse_data$ref_start, reverse_data$ref_end)
reverse_data_long <- reverse_data %>%
  unnest(base_x) %>%
  anti_join(mismatch_positions, by = c("base_x" = "visual_x", "y" = "y"))

## use column align_state to determine the fill color and border color.
base_df <- base_df %>%
  mutate(
    fill_color = case_when(
      align_state %in% c("REF", "MISMATCH", "SOFT_CLIPPED", "DELETION") ~ config$visual$colors$base[base],
      align_state == "MATCH" ~ config$visual$colors$state["MATCH"],
      align_state == "INSERTION" ~ config$visual$colors$state["INSERTION"],
      TRUE ~ "black"
    ),
    border_color = case_when(
      align_state %in% c("HARD_CLIPPED", "SOFT_CLIPPED", "REF", "DELETION") ~ config$visual$colors$border[align_state],
      TRUE ~ "transparent"
    ),
    tile_width = config$visual$tiles$width[align_state],
    tile_height = config$visual$tiles$height[align_state],
    tile_border = config$visual$tiles$border[align_state]
  )

# Check mutation position for multiple bases
mut_color <- "blue"
mut_line_type <- "dashed"
ordered_y <- as.character(sort(unique(base_df$y)))

insert_after_multiple_targets <- function(vec, targets) {
  unlist(lapply(vec, function(x) if (x %in% targets) c(x, x + 1) else x))
}

# 确保 -2 在 ordered_y 中的第一个
ordered_y <- unique(c("-2", ordered_y))

if (!is.character(mut_start)) {
    # Get mutation bases with their counts and sort by frequency (descending)
    mut_bases <- base_df %>%
      filter(visual_x == mut_start - start_axis) %>%
      count(base) %>%
      arrange(desc(n)) %>%  
      pull(base)
  
    unique_bases_count <- length(mut_bases)
  
    if (unique_bases_count >= 2) {
      mut_color <- "red"
      mut_line_type <- "solid"
    
    # Get y positions ordered by mut_bases frequency, keeping read_id together
    ordered_y <- base_df %>%
      left_join(read_df %>% select(read_id, reference_start), 
                by = "read_id") %>%
      filter(visual_x == mut_start - start_axis) %>%
      arrange(match(base, mut_bases), reference_start) %>%
      pull(y)

    # 确保 -2 在 ordered_y 中的第一个
    ordered_y <- c(-2, ordered_y[ordered_y != -2])

    insertion_base <- filter(base_df, align_state == "INSERTION")

    ordered_y <- insert_after_multiple_targets(ordered_y, insertion_base$y - 1)
    
    print(ordered_y)

    # Modify base color and reorder y positions
    base_df <- base_df %>%
      mutate(
        fill_color = ifelse(visual_x == mut_start - start_axis, 
                           config$visual$colors$base[base], 
                           fill_color)
      )
  } else {
    message("Only one base type, no reordering needed")
  }
  
  print(paste("Mutation bases (least frequent first):", paste(mut_bases, collapse = ", ")))
} else {
  # If no mutation position specified, keep original order
  base_df$y <- factor(base_df$y, levels = ordered_y)
}

## parsing snp info
phsable_data_grouped <- data.frame()
if ("snp" %in% colnames(read_df)) {
  # Add is_phsable column to read_df, handling cases where snp column might not exist
  read_df <- read_df %>%
    mutate(is_phsable = if ("snp" %in% colnames(read_df)) {
      grepl("LOCALPAIRED", read_type) | grepl("\\{'paired_hsnp_allele': '[^']+'\\}", snp)
    } else {
      grepl("LOCALPAIRED", read_type)
    })
  read_df <- read_df %>%
    group_by(query_name) %>% 
    mutate(paired_hsnp_allele = if (any(str_detect(snp, ".*'paired_hsnp_allele': '([A-Z])'.*"), na.rm = TRUE)) {
      # 获取匹配的 snp 值
      non_na_snps <- snp[str_detect(snp, ".*'paired_hsnp_allele': '([A-Z])'.*") & !is.na(snp)]
      # 从第一个匹配的 snp 中提取字母
      allele <- first(str_match(non_na_snps, ".*'paired_hsnp_allele': '([A-Z])'.*")[, 2])
      allele
    } else {
      NA_character_
    }) %>%
    replace_na(list(paired_hsnp_allele = "NA"))
  
  phsable_data = subset(read_df, is_phsable)
  
  if (all(is.na(phsable_data$snp))) {
    phsable_data_grouped <- data.frame()
  } else {
    phsable_data_grouped <- phsable_data %>% 
      group_by(query_name) %>% 
      summarize(
        xmin = min(reference_start) - 0.45,
        xmax = max(reference_end) + 0.45 - 1,
        ymin = min(y_position) - 0.45,
        ymax = max(y_position) + 0.45,
        paired_hsnp_allele = if (any(str_detect(snp, ".*'paired_hsnp_allele': '([A-Z])'.*"), na.rm = TRUE)) {
          non_na_snps <- snp[str_detect(snp, ".*'paired_hsnp_allele': '([A-Z])'.*") & !is.na(snp)]
          first(non_na_snps)
        } else {
          NA_character_
        }
      )
    
    phsable_data_grouped$visual_x = phsable_data_grouped$xmin
    phsable_data_grouped$y = phsable_data_grouped$ymin
    
    # Modified this section to handle the type conversion more safely
    phsable_data_grouped <- phsable_data_grouped %>%
      mutate(
        paired_hsnp_allele = as.character(paired_hsnp_allele),  # Ensure character type
        paired_hsnp_allele = case_when(
          is.na(paired_hsnp_allele) ~ "NA",
          str_detect(paired_hsnp_allele, "'paired_hsnp_allele': '([A-Z])'") ~ 
            str_extract(paired_hsnp_allele, "(?<='paired_hsnp_allele': ')([A-Z])(?=')"),
          TRUE ~ "NA"
        )
      )
  }
}

base_plot <- ggplot(data = base_df, aes(x = visual_x, y = as.character(y))) +
  geom_tile(aes(fill = fill_color,
                color = border_color,
                width = tile_width,
                height = tile_width,
                # alpha = base_quality_binary,
                linewidth = I(tile_border)
  ),
  stat = "identity",
  show.legend = TRUE) +
  geom_text(data = subset(base_df, align_state %in% c("MISMATCH", "REF", "INSERTION", "DELETION")),
            aes(label = base),size = config$visual$font$size, color = config$visual$font$color) +
  geom_point(data = subset(base_df, align_state %in% c("DELETION")),
             shape = 95,size = config$visual$font$size, color = "#1e1e1e") +
  {if (!is.character(mut_start) && !is.character(mut_end)) 
    geom_vline(xintercept = c(mut_start - 0.5 - start_axis, mut_end - 0.5 - start_axis), 
               color = mut_color, 
               linetype = mut_line_type, 
               linewidth = 0.2)
  } +
  # scale_y_discrete(expand = expansion(mult = c(.001, .001))) +
  # scale_x_continuous(expand = expansion(mult = c(.001, .001))) +
  scale_y_discrete(expand = c(0,0), breaks = as.character(ordered_y), limits = as.character(ordered_y)) +
  scale_x_continuous(expand = c(0,0), breaks = c(range_x[1]:range_x[2]), labels = (c(range_x[1]:range_x[2])+start_axis)) +
  scale_linewidth_identity() + 
  scale_fill_identity() +
  scale_color_identity() +
  scale_alpha_manual(name = "Base Quality \n (transparency)",
    values = c(0.5,1),
                     breaks = c("<30", ">=30"),
                     na.translate = FALSE)+
  # coord_equal(ratio = 1) +  # 确保1:1的比例
  # theme_void() +
  theme(
    panel.border = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    # axis.ticks.x = element_blank(),
    # axis.text.x = element_blank(),
    # axis.title.x = element_blank(),
    axis.text.x = element_text(size = config$visual$font$size*.pt, angle = 90, vjust = 0.5),
    axis.title.x = element_text(size = config$visual$font$size*.pt*1.5),
    axis.line.y.left = element_blank(),
    axis.ticks.y.left = element_blank(),
    axis.text.y.left = element_blank(),
    axis.title.y.left = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    axis.line.y = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.margin = margin(0, 0, 0, 0, "mm"),
    legend.margin = margin(0, 0, 0, 0, "mm"),
    legend.key.size = unit(3, "mm"),
    legend.title = element_text(size = config$visual$font$size*.pt*1.5),
    legend.text = element_text(size = config$visual$font$size*.pt)
  )

if (nrow(phsable_data_grouped) > 0) {
  base_plot <- base_plot + 
    geom_rect(
      data = phsable_data_grouped,
      aes(
        xmin = xmin, 
        xmax = xmax, 
        ymin = ymin, 
        ymax = ymax,
        color = config$visual$colors$base[paired_hsnp_allele]
      ),
      fill = NA,
      linewidth = config$visual$annotation$phasing_line_width
    )
}

print(phsable_data_grouped)

# Plot calculation function ----
# size of x axis height
get_grob_dimension <- function(gt, grob_name, dimension = c("width", "height")) {
  
  # 匹配参数
  dimension <- match.arg(dimension)
  
  # 获取目标组件索引
  idx <- which(gt$layout$name == grob_name)
  if (length(idx) == 0) {
    message("未找到组件: ", grob_name)
    return(NA_real_)
  }
  
  # 提取行列范围
  if (dimension == "width") {
    cols <- unlist(lapply(idx, function(i) {
      gt$layout$l[i]:gt$layout$r[i]
    }))
    total_dim <- sum(sapply(cols, function(c) {
      convertWidth(gt$widths[c], "mm", valueOnly = TRUE)
    }))
  } else {
    rows <- unlist(lapply(idx, function(i) {
      gt$layout$t[i]:gt$layout$b[i]
    }))
    total_dim <- sum(sapply(rows, function(r) {
      convertHeight(gt$heights[r], "mm", valueOnly = TRUE)
    }))
  }
  
  round(total_dim, 2)  # 保留两位小数
}


# Plot export ----
pb <- ggplot_build(base_plot)
gt <- ggplot_gtable(pb)

# 底下的 x 坐标轴
# get_grob_dimension(gt, "axis-b", "height")
# 
# 底下的 x 坐标轴名字
# get_grob_dimension(gt, "xlab-b", "height")
# 
# 左边 y 坐标轴 
# get_grob_dimension(gt, "axis-l", "width")

# 右边图例的宽度
# get_grob_dimension(gt, "guide-box-right", "width")

# 右边图例的高度
# get_grob_dimension(gt, "guide-box-right", "height")

x_axis_height <- get_grob_dimension(gt, "axis-b", "height") + get_grob_dimension(gt, "xlab-b", "height")

y_axis_width <- get_grob_dimension(gt, "axis-l", "width") + get_grob_dimension(gt, "ylab-l", "width")
guide_width <- get_grob_dimension(gt, "guide-box-right", "width")

# gt$widths[10] <- unit(5, "points")

# 图例和主图之间的间距
pdf(NULL)
legend_gap_width = convertWidth(gt$widths[10], "mm", valueOnly = TRUE)
dev.off()

adjusted_width = (range_x[2] - range_x[1] + 1)  + y_axis_width + legend_gap_width + guide_width

adjusted_height = (range_y[2] - range_y[1] + 1) + x_axis_height

options(ragg.max_dim = 3000000000L)  # 设置为300000万像素，可根据需要调整

ggsave(output_file,
       plot = base_plot,
       device = ragg::agg_png,
       width = adjusted_width,
       height = adjusted_height,
       units = "mm",
       limitsize = FALSE,
       dpi = 300)
# ggsave("examples/simple_base_mismatch.png",
#        plot = base_plot,
#        width = plot_dimensions[1],
#        height = plot_dimensions[2],
#        units = "mm",
#        dpi = 300)
# ggsave("examples/simple_base_mismatch.svg",
#        plot = base_plot,
#        width = plot_dimensions[1],
#        height = plot_dimensions[2],
#        units = "mm",
#        dpi = 300)
