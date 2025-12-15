import { BarChartVariant } from "../types";
export declare const BAR_WIDTH = 16;
/**
 * This function returns the width of the data in the chart, used for padding calculation, scroll amount calculation, and
 * for the width of the chart container.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 */
declare const getWidthOfData: (data: Array<Record<string, string | number>>, categoryKey: string, variant: BarChartVariant) => number;
/**
 * This function returns the padding for the chart, used for the padding of the chart container.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param containerWidth - The width of the container of the chart.
 * @param variant - The variant of the chart.
 */
declare const getPadding: (data: Array<Record<string, string | number>>, categoryKey: string, containerWidth: number, variant: BarChartVariant) => {
    left: number;
    right: number;
};
/**
 * This function returns the scroll amount for the chart, used for the scroll amount of the chart.
 * This can also be used to calculate the width of each group/category.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 */
declare const getWidthOfGroup: (data: Array<Record<string, string | number>>, categoryKey: string, variant: BarChartVariant) => number;
/**
 * This function returns the snap positions for the chart, used for the snap positions of the chart.
 * @param data - The data to be displayed in the chart.
 * @param categoryKey - The key of the category to be displayed in the chart.
 * @param variant - The variant of the chart.
 * @returns The snap positions for the chart.
 */
declare const getSnapPositions: (data: Array<Record<string, string | number>>, categoryKey: string, variant: BarChartVariant) => number[];
export { getPadding, getSnapPositions, getWidthOfData, getWidthOfGroup };
//# sourceMappingURL=BarChartUtils.d.ts.map