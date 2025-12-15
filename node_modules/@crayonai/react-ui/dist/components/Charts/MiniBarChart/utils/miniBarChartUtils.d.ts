import { type MiniBarChartData } from "../types";
export declare const MINI_BAR_WIDTH: number;
export declare const MINI_ELEMENT_SPACING: number;
/**
 * Calculates the left and right padding for the chart container based on available space.
 * If the chart data exceeds the container width, no padding is applied.
 *
 * @param data - The mini bar chart data array
 * @param containerWidth - The total width of the container in pixels
 * @returns An object with left and right padding values in pixels
 */
export declare const getPadding: (data: MiniBarChartData, containerWidth: number) => {
    left: number;
    right: number;
};
/**
 * Filters the data to include only the most recent items that can fit within the container width.
 * This function ensures the chart displays the latest data when space is limited.
 *
 * @param data - The complete mini bar chart data array
 * @param containerWidth - The total width of the container in pixels
 * @returns A filtered array containing only the most recent data items that fit in the container
 */
export declare const getRecentDataThatFits: (data: MiniBarChartData, containerWidth: number) => MiniBarChartData;
type ChartData = Array<{
    value: number;
    label: string;
}>;
/**
 * Transforms the mini bar chart data into a standardized format for rendering.
 * Handles both numeric values and objects with value/label properties.
 *
 * @param data - The mini bar chart data array (can contain numbers or objects with value/label)
 * @returns An array of chart data objects with value and label properties
 */
export declare const transformDataForChart: (data: MiniBarChartData) => ChartData;
export declare const DATA_KEY = "value";
export {};
//# sourceMappingURL=miniBarChartUtils.d.ts.map