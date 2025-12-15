import { format } from "date-fns";
export const formatDateRange = (range) => {
    if (!range)
        return "Select a range";
    const { from, to } = range;
    if (!from)
        return "Select a range";
    if (!to) {
        return format(from, "MMM d, yyyy");
    }
    if (from.toDateString() === to.toDateString()) {
        return format(from, "MMM d, yyyy");
    }
    return `${format(from, "MMM d, yyyy")} - ${format(to, "MMM d, yyyy")}`;
};
export const formatSingleDate = (date) => {
    if (!date)
        return "Select a date";
    return format(date, "MMM d, yyyy");
};
export const formatMultipleDates = (dates) => {
    if (!dates)
        return "Select dates";
    return dates.map((date) => format(date, "MMM d, yyyy")).join(", ");
};
//# sourceMappingURL=DatePickerUtils.js.map