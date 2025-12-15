import { MiniAreaChartData } from "../../MiniAreaChart/types";
import { MiniLineChartData } from "../../MiniLineChart/types";
export declare const MINI_ELEMENT_SPACING: number;
type ChartData = Array<{
    value: number;
    label: string;
}>;
export type MiniChartData = MiniAreaChartData | MiniLineChartData;
/**
 * Transforms mini chart data into a standardized format for rendering.
 * Handles both numeric values and objects with value/label properties.
 * Works for both MiniAreaChart and MiniLineChart components.
 *
 * @param data - The mini chart data array (can contain numbers or objects with value/label)
 * @returns An array of chart data objects with value and label properties
 */
export declare const transformDataForChart: (data: MiniChartData) => ChartData;
/**
 * Filters data to include only the most recent items that can fit within the container width.
 * This function ensures the chart displays the latest data when space is limited.
 * Works for both MiniAreaChart and MiniLineChart components.
 *
 * @param data - The complete mini chart data array
 * @param containerWidth - The total width of the container in pixels
 * @returns A filtered array containing only the most recent data items that fit in the container
 */
export declare const getRecentDataThatFits: (data: MiniChartData, containerWidth: number) => MiniChartData;
export declare const DATA_KEY = "value";
export {};
//# sourceMappingURL=MiniAreaAndLineUtils.d.ts.map