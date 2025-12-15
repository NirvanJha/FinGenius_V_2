import { PieChartData } from "../types";
export interface ChartDimensions {
    outerRadius: number;
    innerRadius: number;
}
export interface TwoLevelChartDimensions extends ChartDimensions {
    middleRadius: number;
}
export interface HoverStyles {
    opacity: number;
    stroke: string;
    strokeWidth: number;
}
export interface ChartHoverHook {
    activeIndex: number | null;
    handleMouseEnter: (event: any, index: number) => void;
    handleMouseLeave: () => void;
}
export interface AnimationConfig {
    isAnimationActive: boolean;
    animationBegin: number;
    animationDuration: number;
    animationEasing: "ease" | "ease-in" | "ease-out" | "ease-in-out" | "linear";
}
/**
 * Calculates the percentage value of a number relative to a total
 * @param value - The value to calculate percentage for
 * @param total - The total value to calculate percentage against
 * @returns The calculated percentage rounded to 2 decimal places
 */
declare const calculatePercentage: (value: number, total: number) => number;
/**
 * Calculates dimensions for standard pie/donut charts
 * @param width - The container width
 * @param variant - The chart variant ('pie' or 'donut')
 * @returns Object containing outer and inner radius values
 */
declare const calculateChartDimensions: (width: number, variant: string) => ChartDimensions;
/**
 * Calculates dimensions for two-level pie charts
 * @param width - The container width
 * @returns Object containing outer, middle, and inner radius values
 */
declare const calculateTwoLevelChartDimensions: (width: number) => TwoLevelChartDimensions;
/**
 * Generates hover style properties for chart cells
 * @param index - The index of the current cell
 * @param activeIndex - The index of the currently hovered cell
 * @returns Object containing hover style properties
 */
declare const getHoverStyles: (index: number, activeIndex: number | null) => HoverStyles;
/**
 * Transforms data by adding percentage calculations
 * @param data - The input data array
 * @param dataKey - The key to use for value calculations
 * @returns Transformed data with added percentage and original value
 */
declare const transformDataWithPercentages: <T extends PieChartData>(data: T, dataKey: keyof T[number]) => {
    percentage: number;
    originalValue: string | number | undefined;
}[];
/**
 * Custom hook for managing chart hover effects
 * @returns Object containing hover state and handlers
 */
declare const useChartHover: () => ChartHoverHook;
/**
 * Creates animation configuration for pie chart
 * @param config - Animation configuration options
 * @returns Animation configuration object
 */
declare const createAnimationConfig: (config?: Partial<AnimationConfig>) => AnimationConfig;
/**
 * Creates event handlers for pie chart
 * @param onMouseEnter - Mouse enter handler
 * @param onMouseLeave - Mouse leave handler
 * @param onClick - Click handler
 * @returns Object containing event handlers
 */
declare const createEventHandlers: (onMouseEnter?: (data: any, index: number) => void, onMouseLeave?: () => void, onClick?: (data: any, index: number) => void) => {
    onMouseEnter: ((data: any, index: number) => void) | undefined;
    onMouseLeave: (() => void) | undefined;
    onClick: ((data: any, index: number) => void) | undefined;
};
export { calculateChartDimensions, calculatePercentage, calculateTwoLevelChartDimensions, createAnimationConfig, createEventHandlers, getHoverStyles, transformDataWithPercentages, useChartHover, };
//# sourceMappingURL=PieChartUtils.d.ts.map