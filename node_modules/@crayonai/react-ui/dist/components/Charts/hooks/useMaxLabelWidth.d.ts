/**
 * Calculates the maximum width of all labels in the chart data.
 *
 * This hook measures the pixel width of each category label and returns
 * the maximum width found. It uses a canvas context to accurately measure
 * text dimensions based on the current theme's font settings.
 *
 * @param data - The chart data array
 * @param categoryKey - The key in the data object that contains the label text
 * @returns The maximum label width in pixels
 */
export declare const useMaxLabelWidth: <T extends Record<string, any>>(data: T[], categoryKey: keyof T) => number;
//# sourceMappingURL=useMaxLabelWidth.d.ts.map