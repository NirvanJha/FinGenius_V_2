interface AngleCalculationResult {
    angle: number;
    height: number;
}
/**
 * Calculates the optimal rotation angle and height for X-axis labels using trigonometry.
 *
 * This hook uses the Pythagorean theorem to determine the optimal angle for rotating
 * labels based on the maximum label width and available horizontal space.
 *
 * Mathematical approach:
 * - Hypotenuse = maxLabelWidth (the label length)
 * - Base = X_AXIS_PADDING (available horizontal space from left edge to first label)
 * - Height = sqrt(hypotenuse² - base²)
 * - Angle = atan(height / base) converted to degrees
 *
 * @param maxLabelWidth - The maximum width of all labels in pixels
 * @param enabled - Whether to calculate the angle (typically based on tickVariant)
 * @returns Object containing the calculated angle (in degrees) and required height
 */
export declare const useAutoAngleCalculation: (maxLabelWidth: number, enabled: boolean, widthOfData?: number) => AngleCalculationResult;
export {};
//# sourceMappingURL=useAutoAngleCalculation.d.ts.map