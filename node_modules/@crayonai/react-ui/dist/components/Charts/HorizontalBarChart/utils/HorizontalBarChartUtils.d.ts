import { HorizontalBarChartVariant } from "../types";
export declare const BAR_HEIGHT = 16;
export declare const BAR_GAP = 10;
/**
 * This function returns the height of the data in the chart, used for padding calculation, scroll amount calculation, and
 * for the height of the chart container.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 * @param labelHeight - The height of the category label.
 */
declare const getHeightOfData: (data: Array<Record<string, string | number>>, categoryKey: string, variant: HorizontalBarChartVariant, labelHeight: number) => number;
/**
 * This function returns the padding for the chart, used for the padding of the chart container.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param containerHeight - The height of the container of the chart.
 * @param variant - The variant of the chart.
 * @param labelHeight - The height of the category label.
 */
declare const getPadding: (data: Array<Record<string, string | number>>, categoryKey: string, containerHeight: number, variant: HorizontalBarChartVariant, labelHeight: number) => {
    top: number;
    bottom: number;
};
/**
 * This function returns the height of each group/category.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 * @param labelHeight - The height of the category label.
 */
declare const getHeightOfGroup: (data: Array<Record<string, string | number>>, categoryKey: string, variant: HorizontalBarChartVariant, labelHeight: number) => number;
/**
 * This function returns the snap positions for the chart, used for the snap positions of the chart.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 * @param labelHeight - The height of the category label.
 * @returns The snap positions for the chart.
 */
declare const getSnapPositions: (data: Array<Record<string, string | number>>, categoryKey: string, variant: HorizontalBarChartVariant, labelHeight: number) => number[];
export { getHeightOfData, getHeightOfGroup, getPadding, getSnapPositions };
//# sourceMappingURL=HorizontalBarChartUtils.d.ts.map