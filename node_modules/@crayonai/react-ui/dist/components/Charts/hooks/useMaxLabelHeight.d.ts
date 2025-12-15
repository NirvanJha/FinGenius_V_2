import { XAxisTickVariant } from "../types";
export declare const useMaxLabelHeight: (data: Record<string, string | number>[], categoryKey: string, tickVariant: XAxisTickVariant, widthOfGroup?: number) => number;
/**
 * Hook to calculate the maximum label height for horizontal bar charts.
 * This considers the vertical layout and label width constraints specific to horizontal bars.
 * @param data - The chart data
 * @param categoryKey - The category key
 * @param labelWidth - The available width for labels (full chart width for horizontal bars)
 * @returns The calculated label height
 */
export declare const useHorizontalBarLabelHeight: (data: Record<string, string | number>[], categoryKey: string, labelWidth: number) => number;
//# sourceMappingURL=useMaxLabelHeight.d.ts.map