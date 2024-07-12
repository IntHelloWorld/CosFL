public enum ErrorFormat {
  LEGACY {
    @Override
    public MessageFormatter toFormatter(
        SourceExcerptProvider source, boolean colorize) {
      VerboseMessageFormatter formatter = new VerboseMessageFormatter(source);
      formatter.setColorize(colorize);
      return formatter;
    }
  },
  SINGLELINE {
    @Override
    public MessageFormatter toFormatter(
        SourceExcerptProvider source, boolean colorize) {
      LightweightMessageFormatter formatter = new LightweightMessageFormatter(
          source);
      formatter.setColorize(colorize);
      return formatter;
    }
  };

  /**
   * Convert to a concrete formatter.
   */
  public abstract MessageFormatter toFormatter(
      SourceExcerptProvider source, boolean colorize);
}