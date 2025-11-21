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
start_axis <- if (length(args) >= 8) 0 else 0

plot_alignment <- function(base_df_path, read_df_path, reference_df_path, output_file,  mut_start, mut_end, mut_type, start_axis) {
  print(start_axis)
  
  # 处理可能为 None 的参数
  mut_start <- if (is.null(mut_start) || mut_start == "None") NA else as.numeric(mut_start)
  mut_end <- if (is.null(mut_end) || mut_end == "None") NA else as.numeric(mut_end)
  mut_type <- if (is.null(mut_type) || mut_type == "None") NA else mut_type
  
  # Read the CSV files
  base_df <- read.csv(base_df_path)
  read_df <- read.csv(read_df_path)
  reference_df <- read.csv(reference_df_path)
  
  # Your existing plotting code here
  library(dplyr)
  library(aplot)
  library(ggplot2)
  library(grid)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggfittext)
  library(ggnewscale)
  
  ## import data ----
  # Read the CSV files
  base_data <- read_csv(base_df_path, show_col_types = FALSE)
  read_data <- read_csv(read_df_path, show_col_types = FALSE)
  ref_data <- read_csv(reference_df_path, show_col_types = FALSE)
  
  ## color setting ----
  BASE_COLORS <- c(
    'A' = '#33a02c', 
    'T' = '#1f78b4',
    'G' = '#fccde5',
    'C' = '#fdb462',
    'N' = '#444444',
    '-' = '#ffffff'
  )
  
  ## size setting ----
  base_tile_height = 0.75
  base_tile_width = 0.75
  
  deletion_height = 0.2
  deletion_width = 0.38
  
  clip_linewidth = 0.18
  str_linewidth = 0.2
  
  read_point_size = 0.3
  reverse_point_size = 0.6
  annotate_line_width = 0.1
  phsing_line_width = 0.2
  mutation_reads_width = 0.5
  deletion_linewidth = 0.05
  
  text_margin = 0.15
  
  min_font_size = 0
  
  range_x = range(ref_data$ref_pos)
  range_y = range(base_data$y)
  
  ## is reverse data
  reverse_data = data.frame(
    read_id = read_data$read_id,
    y = read_data$y_position,
    ref_start = read_data$reference_start + 1,
    ref_end = read_data$reference_end - 1,
    is_reverse = read_data$is_reverse
  )
  
  # 获取错配位置
  mismatch_positions <- base_data %>%
    filter(align_state == "MISMATCH") %>%
    select(visual_x, y) %>%
    distinct()

  # 展开序列并过滤掉错配位置
  reverse_data$base_x <- Map(seq, reverse_data$ref_start, reverse_data$ref_end)
  reverse_data_long <- reverse_data %>%
    unnest(base_x) %>%
    anti_join(mismatch_positions, by = c("base_x" = "visual_x", "y" = "y"))

  base_data$base <- factor(base_data$base, levels = c('A', 'T', 'G', 'C', 'N', '-'))

  ## plot reads----
  p_mismatch <- ggplot(data = base_data, aes(x = visual_x, y = y)) + 
    geom_segment(
      data = subset(base_data, align_state == "MISMATCH"),
      aes(x = visual_x, xend = visual_x,
          y = y, yend = -1),
      color = "grey60",
      linewidth = 0.1,
      alpha = 0.38
    ) +
    geom_tile(
      data = subset(base_data, align_state == "MISMATCH"),
      aes(fill = base,
          alpha = ifelse(base_quality < 20, 0.5, 1)),
      height = base_tile_height,
      width = base_tile_width, show.legend = FALSE
    ) +
    theme_void() +
    scale_fill_manual(values = BASE_COLORS) +
    scale_alpha_identity()  # Use the alpha values as-is
  
  ## basic plot layers ----
  p_match <- p_mismatch + 
    geom_tile(data = subset(base_data, align_state %in% "MATCH"), fill = "#d9d9d9",height = base_tile_height,width = base_tile_width,
              aes(alpha = ifelse(base_quality < 20, 0.5, 1)), show.legend = FALSE) +
    geom_point(data = reverse_data_long, aes(x = base_x, y = y, shape = is_reverse, colour = is_reverse), 
               size = reverse_point_size, show.legend = FALSE)+
        scale_shape_manual(values = c("FALSE" = ">", "TRUE" = "<")) +
        scale_color_manual(values = c('FALSE' = '#60a7d8', 'TRUE' = '#be75ac')) +
    new_scale_color() +
    geom_fit_text(data = subset(base_data, align_state %in% c("MISMATCH")),
                  aes(label = base),
                  color = "#130f30",min.size = min_font_size,
                  padding.x = grid::unit(text_margin, "mm"),
                  padding.y = grid::unit(text_margin, "mm")
    )
    # # print text for debug
    # geom_fit_text(data = subset(base_data, align_state %in% "MATCH"),
    #               aes(label = base),
    #               color = "#130f30",min.size = min_font_size,
    #               padding.x = grid::unit(text_margin, "mm"),
    #               padding.y = grid::unit(text_margin, "mm")
    # )
  
  p_match_insert <- p_match + 
    geom_tile(data = subset(base_data, align_state == "INSERTION"), 
              fill = "#54278f",height = base_tile_height,width = base_tile_width,
              aes(alpha = ifelse(base_quality < 20, 0.5, 1)), show.legend = FALSE) + 
    geom_fit_text(data = subset(base_data, align_state %in% c("INSERTION")),
                aes(label = base),
                color = "white",min.size = min_font_size,
                padding.x = grid::unit(text_margin, "mm"),
                padding.y = grid::unit(text_margin, "mm")
    )
  
  ## add deltion to the bases is DELETION
  p_deletion = p_match_insert + geom_tile(data = subset(base_data, align_state == "DELETION"), aes(fill = base),color = '#130f30',height=base_tile_height,width=base_tile_width,linewidth = deletion_linewidth) +
    geom_tile(data = subset(base_data, align_state == "DELETION"), fill = '#130f30', color = '#130f30',height=deletion_height,width=deletion_width)+
    theme_void() +
    scale_fill_manual(values = BASE_COLORS)
  
  ## add border to the bases is CLIP
  p_clip <- p_deletion + 
    geom_tile(data = subset(base_data, align_state %in% c("HARD_CLIPPED", "SOFT_CLIPPED")),
              aes(fill = base,
                  alpha = 0.5),color = '#130f30',
              height=base_tile_height,width=base_tile_width, linewidth = clip_linewidth, show.legend = FALSE) +
    scale_fill_manual(values = BASE_COLORS) +
    theme_void()
  
  ## plot reference ----
  # set a kind of theme for reference plot
  p_ref <- ggplot(data = ref_data, aes(x = ref_pos, y = -1)) +
    geom_tile(aes(fill = base), color='#130f30', show.legend = FALSE) +
    geom_fit_text(aes(label = base), color = '#130f30',min.size = min_font_size,
                  padding.x = grid::unit(text_margin, "mm"),
                  padding.y = grid::unit(text_margin, "mm")) +
    labs(x = "Reference Position")+
    scale_fill_manual(values = BASE_COLORS,na.value = "white") +
    theme_void() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.text.x = element_text(size = 6, angle = 45, hjust = 0.5, vjust = 0.5),
      axis.ticks.x = element_line(),
      axis.ticks.length = unit(3, "mm")
    ) +
    scale_x_continuous(expand = c(0,0),limits = c(range_x[1] - 1, range_x[2] + 1),
                       breaks = seq(floor(range_x[1]), ceiling(range_x[2]), by = 5))
  
  ## str region ----
  if ("str" %in% colnames(base_data)) {
    # Rename the column temporarily
    base_data <- base_data %>%
      rename(str_col = str) %>%
      mutate(is_str = grepl("'?is_str'?\\s*:\\s*True", str_col, ignore.case = TRUE)) %>%
      rename(str = str_col)

    p_str <- p_clip + 
      geom_tile(data = subset(base_data, is_str == TRUE), 
                fill = 'transparent',
                color = "#130f30",
                height=base_tile_height,width=base_tile_width,linewidth = str_linewidth, show.legend = FALSE)
  } else {
    p_str <- p_clip
  }
  
  ## calculate features ----
  ## mean_base_q
  mean_base_q <- base_data %>%
    group_by(read_id) %>%
    summarize(mean_base_quality = mean(base_quality, na.rm = TRUE))
  
  ## join the mean_base_q to the read_data
  read_data <- read_data %>%
    left_join(mean_base_q, by = "read_id")
  
  ## is_phsable
  # Add is_phsable column to read_data, handling cases where snp column might not exist
  read_data <- read_data %>%
    mutate(is_phsable = if ("snp" %in% colnames(read_data)) {
      grepl("LOCALPAIRED", read_type) | grepl("\\{'paired_hsnp_allele': '[^']+'\\}", snp)
    } else {
      grepl("LOCALPAIRED", read_type)
    })
    
  ## parsing snp info
  if ("snp" %in% colnames(read_data)) {
    read_data <- read_data %>%
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
  }
  
  read_data$visual_x = read_data$reference_start
    
  ## group phsable data ----
  # Group phsable_data by y_position and calculate xmax for each group
  phsable_data = subset(read_data, is_phsable)
  
  if ("snp" %in% colnames(read_data)) { 
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

    p_str_aano <- p_str + 
      geom_point(data = read_data, aes(x = visual_x - 0.5, y = y_position, colour = paired_hsnp_allele), size = read_point_size)

    if (nrow(phsable_data_grouped) > 0) {
      p_str_aano <- p_str_aano + 
        geom_rect(
        data = phsable_data_grouped,
        aes(
          xmin = xmin, 
          xmax = xmax, 
          ymin = ymin, 
          ymax = ymax,
          color = paired_hsnp_allele
        ),
        fill = NA,
        linewidth = phsing_line_width
      )
    }
  } else {
    phsable_data_grouped <- data.frame()
    p_str_aano <- p_str + 
      geom_point(data = read_data, aes(x = visual_x - 0.5, y = y_position), color = '#130f30', size = read_point_size)

    if (nrow(phsable_data_grouped) > 0) {
      p_str_aano <- p_str_aano + 
        geom_rect(
        data = phsable_data_grouped,
        aes(
          xmin = xmin, 
          xmax = xmax, 
          ymin = ymin, 
          ymax = ymax
        ),
        color = 'black',
        fill = NA,
        linewidth = phsing_line_width
      )
    }
  }
  
  p_str_aano <- p_str_aano +
    scale_x_continuous(
      expand = c(0,0),
      limits = c(range_x[1] - 1, range_x[2] + 1),
      breaks = seq(floor(range_x[1]), ceiling(range_x[2]), by = 1)  # 添加这行来确保 x 轴的刻度
    ) +
    scale_y_continuous(
      expand = c(0,0),
      limits = c(range_y[1] - 1, range_y[2] + 1),
      breaks = seq(floor(range_y[1]), ceiling(range_y[2]), by = 1)
    ) +
    scale_color_manual(values = BASE_COLORS, na.value = '#130f30') +
    theme_void() + 
    theme(
      panel.grid.major.y = element_line(color = "grey80", linewidth = 0.1),  # 修改这行来显示所有主网格线
      panel.grid.minor.y = element_blank(),
      # panel.grid.major.x = element_line(color = "grey80", linewidth = 0.1),
      # panel.grid.minor.x = element_line(color = "grey80", linewidth = 0.1),
      panel.border = element_blank(),
      panel.background = element_blank(),
      legend.position = "none"
    )
  
  ## add features of allele_source
  read_data <- read_data %>%
    mutate(allele_source = if ("classify_allelic_reads" %in% colnames(.)) {
      ifelse(
        is.na(classify_allelic_reads), 
        -1, 
        as.numeric(gsub(".*'allele_source': ([0-9]+).*", "\\1", classify_allelic_reads))
      )
    } else {
      -1  # Default value when column doesn't exist
    })
  
  read_data$allele_source <- as.character(read_data$allele_source)
  
  # 添加 geom_hline 图层 for mutation reads
  p_str_aano <- p_str_aano + 
    geom_hline(data = subset(read_data, allele_source == "1"), 
               aes(yintercept = y_position), linewidth = mutation_reads_width,
               color = "#e41a1c")
  
  total_layers <- length(p_str_aano$layers)
  
  p_str_aano$layers <- c(p_str_aano$layers[[total_layers]], p_str_aano$layers[-total_layers])
  
  ## get snp position
  if ("snp" %in% colnames(read_data)) {
    snp_position <- unique(base_data[!is.na(base_data$snp), "ref_pos"]) |> as.numeric()
  
    p_str_aano <- p_str_aano + annotate("rect", 
                          xmin = snp_position - 0.5, 
                          xmax = snp_position + 0.5,
                          ymin = range_y[1] - 0.5, 
                          ymax = range_y[2] + 0.5,
                          color = "blue",
                          fill = "transparent",
                        alpha = 0,
                        linewidth = annotate_line_width) + 
      annotate('text', 
               x = snp_position,
               y = range_y[2] + 0.5,
               label = "hSNP")
  }
  
  # 'mut_refalt_1': "('12', 70620853, 70620853, 'G', 'T')"
  # 'mut_info_1': "(14, 1, 'mismatch', 'G', 'T', 14)",
  # 'mut_res_all_1': "((14, 1, 'mismatch', 'G', 'T', 14), (44, 1, 'mismatch', "
  # "'G', 'T', 44), (46, 2, 'insertion', 'TT', 47))",
  # mut_start = 70620853
  # mut_end = 70620853
  # mut_type = 'mismatch'
  if (!is.na(mut_start) && !is.na(mut_end) && !is.na(mut_type)) {
    p_str_aano <- p_str_aano + annotate("rect", 
                          xmin = mut_start - 1.5, 
                          xmax = mut_end - 0.5,
                          ymin = range_y[1] - 0.5, 
                          ymax = range_y[2] + 0.5,
                          color = "red",
                          fill = "transparent",
                          alpha = 0,
                          linewidth = annotate_line_width) + 
      annotate('text', 
               x = mut_start - 1.5,
               y = range_y[2] + 0.5,
               label = mut_type)
  }
  
  ## plot features ----
  # Normalize reference_start within each y_position to the range 0, 1, 2
  read_data <- read_data %>%
    group_by(y_position) %>%
    arrange(reference_start, .by_group = TRUE) %>%
    mutate(reference_start_norm = as.numeric(factor(reference_start)) - 1)
  
  ## reverse plot pallete
  color_strand <- c("#be75ac", "#60a7d8")
  names(color_strand) <- c(TRUE, FALSE)
  
  color_qcfail <- c("#b2182b", "#e0e0e0")
  names(color_qcfail) <- c(TRUE, FALSE)
  
  color_duplicate <- c("#b2182b", "#e0e0e0")
  names(color_duplicate) <- c(TRUE, FALSE)
  
  color_filter <- c("#b2182b", "#e0e0e0")
  names(color_filter) <- c(TRUE, FALSE)
  
  color_phasable <- c("#238b45", "#e0e0e0")
  names(color_phasable) <- c(TRUE, FALSE)
  
  color_allele_source <- c("#e41a1c", "#e0e0e0","white")
  names(color_allele_source) <- c('1','0','-1')
  
  # Define the theme_features function
  theme_features <- function() {
    theme_void() + 
      theme(
        plot.title = element_text(size = 6, angle = 90),
        panel.border = element_rect(fill = NA, linewidth = 0.5)
      )
  }
  
  ## functional ----
  create_feature_plot_bool <- function(data, feature_column, fill_pallete) {
    # Pivot the data to a long format for the specific feature
    data_long <- data %>%
      select(reference_start_norm, y_position, !!sym(feature_column)) %>%
      pivot_longer(cols = c(!!sym(feature_column)),
                   names_to = "property", values_to = "value")
    
    # Create the tile plot
    plot <- ggplot(data_long, aes(x = reference_start_norm, y = y_position)) +
      geom_tile(aes(fill = as.factor(value)), color = "white", 
                width = 0.95, height = 0.95, show.legend = FALSE) +
      labs(title = feature_column)+
      scale_fill_gradient(na.value = "white") +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0)) +
      theme_features()
    
    # Set color scale based on the feature
    plot <- plot + scale_fill_manual(values = fill_pallete)
    
    return(plot)
  }
  
  # Create plots for is_reverse and is_qcfail
  plot_is_reverse <- create_feature_plot_bool(read_data, "is_reverse",  fill_pallete = color_strand)
  plot_is_qcfail <- create_feature_plot_bool(read_data, "is_qcfail",  fill_pallete = color_qcfail)
  plot_is_duplicate <- create_feature_plot_bool(read_data, "is_duplicate",  fill_pallete = color_duplicate)
  # plot_is_filter <- create_feature_plot_bool(read_data, "is_filter", fill_pallete = color_filter)
  # plot_is_phsable <- create_feature_plot_bool(read_data, "is_phsable", fill_pallete = color_phasable)
  plot_allele_source <- create_feature_plot_bool(read_data, "allele_source", fill_pallete = color_allele_source)
  
  # # Create a list of plots
  plot_list_discrete <- list(is_reverse = plot_is_reverse, is_qcfail = plot_is_qcfail, 
                             is_duplicate = plot_is_duplicate, allele_source = plot_allele_source)
  
  color_mean_base_quality <- c("#f7fcfd", "#99d8c9", "#006d2c")
  names(color_mean_base_quality) <- c("low", "medium", "high")
  
  color_mapping_quality <- c("#f7fcfd", "#99d8c9", "#006d2c")
  names(color_mapping_quality) <- c("low", "medium", "high")
  
  color_query_length <- c("#f7fcfd", "#99d8c9", "#006d2c")
  names(color_query_length) <- c("low", "medium", "high")
  
  create_feature_plot_continuous <- function(data, feature_column, fill_palette) {
    library(scales)
    
    # Pivot the data to a long format for the specific feature
    data_long <- data %>%
      select(reference_start_norm, y_position, !!sym(feature_column)) %>%
      pivot_longer(cols = c(!!sym(feature_column)),
                   names_to = "property", values_to = "value")
    
    # Determine the range of the data
    min_value <- min(data_long$value, na.rm = TRUE)
    max_value <- max(data_long$value, na.rm = TRUE)
    
    # Define a custom rescaling function
    custom_rescaler <- function(x, from) {
      if (diff(from) == 0) {
        # Only one unique value; map it to the middle of the gradient
        return(rep(0.5, length(x)))
      } else {
        # Standard rescaling
        return((x - from[1]) / diff(from))
      }
    }
    
    # Create the tile plot
    plot <- ggplot(data_long, aes(x = reference_start_norm, y = y_position)) +
      geom_tile(aes(fill = value), color = "white", 
                width = 0.95, height = 0.95, show.legend = TRUE) +
      geom_fit_text(aes(label = round(value, 0)),
                    padding.x = grid::unit(text_margin, "mm"),min.size = min_font_size,
                    padding.y = grid::unit(text_margin, "mm")) +
      scale_fill_gradientn(
        colors = fill_palette,
        na.value = "white",
        rescaler = custom_rescaler,
        limits = c(min_value, max_value),
        guide = guide_colorbar(
          title = feature_column,
          title.position = "bottom",
          label.position = "bottom",
          direction = "horizontal"
        )
      ) +
      labs(title = feature_column) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      theme_features() +
      theme(legend.position = "top")
    
    return(plot)
  }
  
  plot_mean_base_quality <- create_feature_plot_continuous(read_data, "mean_base_quality", fill_palette = color_mean_base_quality)
  plot_mapping_quality <- create_feature_plot_continuous(read_data, "mapping_quality", fill_palette = color_mapping_quality)
  plot_query_length <- create_feature_plot_continuous(read_data, "query_alignment_length", fill_palette = color_query_length)
  
  plot_list_continuous <- list(mean_base_quality = plot_mean_base_quality, mapping_quality = plot_mapping_quality, query_length = plot_query_length)
  
  plot_list <- c(plot_list_discrete, plot_list_continuous, plot_query_length)
  
  # display the query name 
  p_query_name <- ggplot(data = read_data, aes(x = reference_start_norm, y = y_position)) +
    geom_tile(fill = 'white')+
    geom_fit_text(aes(label = query_name),min.size = min_font_size,
                  padding.x = grid::unit(0, "mm"),
                  padding.y = grid::unit(0, "mm")) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    theme_features()
  
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
  # p_str_aano_pb <- ggplot_build(p_str_aano)
  # p_str_aano_gt <- ggplot_gtable(p_str_aano_pb)
  
  p_ref_pb <- ggplot_build(p_ref)
  p_ref_gt <- ggplot_gtable(p_ref_pb)
  
  # 底下的 x 坐标轴
  ref_x_axis_height = get_grob_dimension(p_ref_gt, "axis-b", "height")
  
  p_query_length_pb <- ggplot_build(plot_list$query_length)
  p_query_length_gt <- ggplot_gtable(p_query_length_pb)
  
  head_title_height <- get_grob_dimension(p_query_length_gt, "title", "height")
  
  legend_width <- get_grob_dimension(p_query_length_gt, "guide-box-top", "height")
  
  ## plot size config ----
  label_width = 30
  ## max features width
  feature_width <- table(read_data$y_level) |> max()
  
  # max base_plot width
  base_plot_width <- diff(range_x) * 1 
  
  # max base_plot height
  min_base_plot_height = 40 * 1
  base_plot_height <- max(diff(range_y) * 1, min_base_plot_height)
  
  # compute total size
  feature_all_width = feature_width * length(plot_list) * 2
  sub_width = c(feature_all_width, base_plot_width, label_width, legend_width)
  percent_width = prop.table(sub_width)
  total_width = base_plot_width + feature_all_width + label_width + legend_width
  total_height = base_plot_height + head_title_height + 1 + ref_x_axis_height
  
  units = 'mm'
  
  proption_labels = percent_width[3]
  proption_feature = percent_width[1] / length(plot_list) * 1
  
  proption_ref = (1+ref_x_axis_height)/ total_height
  
  # Save the plot
  p <- p_str_aano %>% 
    insert_bottom(p_ref,height = proption_ref) %>%
    insert_right(p_query_name, width = proption_labels) %>%
    insert_left(plot_list$allele_source, width = proption_feature) %>%
    insert_left(plot_list$is_reverse, width = proption_feature) %>%
    insert_left(plot_list$is_qcfail, width = proption_feature) %>% 
    insert_left(plot_list$is_duplicate, width = proption_feature) %>%
    insert_left(plot_list$mean_base_quality, width = proption_feature) %>%
    insert_left(plot_list$mapping_quality, width = proption_feature) %>%
    insert_left(plot_list$query_length, width = proption_feature)
  
  ggsave(output_file, p, width = total_width, height = total_height, 
         units = units,
         dpi = 300, limitsize = FALSE)
}

plot_alignment(base_df_path, read_df_path, reference_df_path, output_file, mut_start, mut_end, mut_type, start_axis)
