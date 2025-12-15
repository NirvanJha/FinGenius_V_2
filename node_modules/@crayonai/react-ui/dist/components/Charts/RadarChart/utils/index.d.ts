/**
 * Truncates text to fit within specified width bounds
 * @param text - The original text
 * @param maxWidth - Maximum width in pixels
 * @param fontSize - Font size for measurement
 * @returns Truncated text with ellipsis if needed
 */
declare const truncateText: (text: string, maxWidth: number, fontSize?: number) => string;
/**
 * Calculates the available width for a text label based on its position and anchor point
 *
 * This function determines how much horizontal space is available for a text label
 * by checking its position relative to the container bounds. The calculation varies
 * based on the text anchor position:
 *
 * - For "start" anchored text: Available space is from labelX to container right edge
 * - For "end" anchored text: Available space is from container left edge to labelX
 * - For "middle" anchored text: Takes minimum of left/right space and doubles it
 *
 * @param labelX - X coordinate of the label position
 * @param containerWidth - Total width of the container
 * @param textAnchor - How text is anchored ("start", "end", or "middle")
 * @param padding - Optional padding to maintain from container edges (default 10px)
 * @returns The maximum width available for the label in pixels
 */
declare const calculateAvailableWidth: (labelX: number, containerWidth: number, textAnchor: string, padding?: number) => number;
export { calculateAvailableWidth, truncateText };
//# sourceMappingURL=index.d.ts.map